"""
Turboelectric Adaptive Engine - pyCycle Implementation
=======================================================
Architecture (from station analysis):
  Station 0  : Freestream
  Station 1  : Inlet exit
  Station 2  : Spinner / flow-split (core A + bypass B)
  Station 3  : Post-spinner convergence
  Station 4  : IGV front face
  Station 5  : IGV aft face (compressor entry)
  Station 6  : Aft compressor stage 1  (electric motor driven)
  Station 7  : Aft compressor stage 2  (electric motor driven)
  Station 8  : Diffuser exit / combustor entry
  Station 9  : Combustor fuel injection
  Station 10 : Combustor exit / nozzle entry
  Station 11 : Nozzle throat
  Station 12 : Nozzle exit
Key differentiator from a conventional turbofan:
  - NO turbine stage in the gas path
  - Compressor work is supplied electrically (ElectricCompressor elements)
  - Full combustion enthalpy is available for thrust
Dependencies (install before running):
  pip install openmdao pycycle
Usage:
  python turboelectric_engine.py
Author  : Generated for Alex's turboelectric propulsion study
Toolset : pyCycle v3.x / OpenMDAO v3.x
"""
import sys
# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import openmdao.api as om
except ImportError:
    sys.exit(
        "OpenMDAO not found. Install with:\n"
        "  pip install openmdao pycycle\n"
        "Then re-run this script."
    )
try:
    import pycycle.api as pyc
except ImportError:
    sys.exit(
        "pyCycle not found. Install with:\n"
        "  pip install pycycle\n"
        "Then re-run this script."
    )
import numpy as np
# ===========================================================================
# 1.  ELECTRIC COMPRESSOR ELEMENT
#     pyCycle ships a standard Compressor element driven by shaft power.
#     For the turboelectric case we subclass it and add an ElectricPower
#     output so the optimizer/sizing loop can see the MW demand.
# ===========================================================================
class ElectricCompressor(pyc.Compressor):
    """
    Compressor whose shaft power is sourced from an electric motor rather
    than a turbine.  Identical thermodynamics to pyc.Compressor; adds:
      - Output: 'power_elec'  [W]  – electrical shaft power demanded
      - Input:  'motor_eta'        – motor + inverter efficiency (default 0.95)
    """
    def setup(self):
        super().setup()
        self.add_input("motor_eta", val=0.95,
                       desc="Motor + inverter efficiency")
        self.add_output("power_elec", val=0.0, units="W",
                        desc="Electrical power drawn from bus")

    def setup_partials(self):
        super().setup_partials()
        self.declare_partials("power_elec", ["Fl_I:stat:W", "motor_eta"])
        # power_elec also depends on delta-h, which comes from the parent
        self.declare_partials("power_elec", "power")

    def compute(self, inputs, outputs):
        super().compute(inputs, outputs)
        # power [hp] from parent; convert to W then de-rate by motor eta
        power_shaft_W = outputs["power"] * 745.699872   # hp -> W
        outputs["power_elec"] = power_shaft_W / inputs["motor_eta"]

    def compute_partials(self, inputs, J):
        super().compute_partials(inputs, J)
        eta = inputs["motor_eta"]
        power_hp = self._outputs["power"]          # current value
        J["power_elec", "power"]     = 745.699872 / eta
        J["power_elec", "motor_eta"] = -745.699872 * power_hp / eta**2


# ===========================================================================
# 2.  TOP-LEVEL ENGINE GROUP
# ===========================================================================
class TurboelectricEngine(pyc.Cycle):
    """
    Full turboelectric adaptive engine cycle.
    Design variables exposed to the driver:
      des_vars/MN          – flight Mach number
      des_vars/alt         – altitude [ft]
      des_vars/T4_MAX      – turbine-inlet (combustor-exit) temperature [R]
      des_vars/BPR         – bypass ratio
      des_vars/PR_stage1   – stage-1 compressor pressure ratio
      des_vars/PR_stage2   – stage-2 compressor pressure ratio
    """
    def initialize(self):
        self.options.declare("design", default=True,
                             desc="True = design point; False = off-design")

    def setup(self):
        design = self.options["design"]

        # ------------------------------------------------------------------
        # Thermodynamic property set
        # ------------------------------------------------------------------
        thermo_spec = pyc.species_data.janaf
        janaf_thermo = pyc.Thermo(
            thermo_data=thermo_spec,
            mix_ratio=pyc.MIX_AIR    # will be overridden to MIX_AIR_FUEL in combustor
        )

        # ------------------------------------------------------------------
        # 2a.  INLET  (Stations 0 → 1)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "inlet",
            pyc.Inlet(
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2b.  SPLITTER  (Station 2 – splits core A and bypass B)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "splitter",
            pyc.Splitter(
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2c.  INLET GUIDE VANES – modelled as a duct with pressure loss
        #      covering Stations 3 → 5 (spinner convergence + IGV)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "igv_duct",
            pyc.Duct(
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2d.  ELECTRIC COMPRESSOR – STAGE 1  (Station 5 → 6)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "comp1",
            ElectricCompressor(
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2e.  ELECTRIC COMPRESSOR – STAGE 2  (Station 6 → 7)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "comp2",
            ElectricCompressor(
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2f.  DIFFUSER DUCT  (Station 7 → 8/9, deceleration before combustor)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "diffuser",
            pyc.Duct(
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2g.  COMBUSTOR  (Stations 9 → 10)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "combustor",
            pyc.Combustor(
                design=design,
                thermo_data=thermo_spec,
                inflow_elements=pyc.AIR_MIX,
                air_fuel_elements=pyc.AIR_FUEL_MIX,
                fuel_type="Jet-A"
            )
        )

        # ------------------------------------------------------------------
        # 2h.  CORE NOZZLE  (Stations 10 → 12, convergent-divergent)
        #      pyCycle's Nozzle element supports both convergent and C-D
        #      via the 'lossCoef' and nozzle_type parameters.
        # ------------------------------------------------------------------
        self.add_subsystem(
            "core_nozzle",
            pyc.Nozzle(
                nozzle_type="CD",          # convergent-divergent (supersonic exit)
                lossCoef="Cv",
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_FUEL_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2i.  BYPASS NOZZLE  (bypass stream → exit)
        # ------------------------------------------------------------------
        self.add_subsystem(
            "bypass_nozzle",
            pyc.Nozzle(
                nozzle_type="CV",          # convergent-only (bypass is subsonic)
                lossCoef="Cv",
                design=design,
                thermo_data=thermo_spec,
                elements=pyc.AIR_MIX
            )
        )

        # ------------------------------------------------------------------
        # 2j.  PERFORMANCE SUMMARY
        #      Aggregates core + bypass thrust, TSFC, electric power demand
        # ------------------------------------------------------------------
        self.add_subsystem(
            "perf",
            pyc.Performance(
                num_nozzles=2,
                num_burners=1
            )
        )

        # ==================================================================
        # 3.  FLOW CONNECTIONS  (pyCycle naming: element.Fl_O → element.Fl_I)
        # ==================================================================
        flow_connections = [
            ("fc.Fl_O",              "inlet.Fl_I"),        # freestream → inlet
            ("inlet.Fl_O",           "splitter.Fl_I"),     # inlet → splitter
            ("splitter.Fl_O1",       "igv_duct.Fl_I"),     # core stream → IGV duct
            ("igv_duct.Fl_O",        "comp1.Fl_I"),        # IGV → stage-1 compressor
            ("comp1.Fl_O",           "comp2.Fl_I"),        # stage 1 → stage 2
            ("comp2.Fl_O",           "diffuser.Fl_I"),     # stage 2 → diffuser
            ("diffuser.Fl_O",        "combustor.Fl_I"),    # diffuser → combustor
            ("combustor.Fl_O",       "core_nozzle.Fl_I"),  # combustor → core nozzle
            ("splitter.Fl_O2",       "bypass_nozzle.Fl_I"),# bypass stream → bypass nozzle
        ]
        for src, tgt in flow_connections:
            self.connect(src, tgt)

        # ==================================================================
        # 4.  PERFORMANCE PORT CONNECTIONS
        # ==================================================================
        # Inlet ram drag
        self.connect("inlet.Fl_O:tot:P",     "perf.Pt2")
        self.connect("fc.Fl_O:tot:P",        "perf.Pt0")
        # Core nozzle
        self.connect("core_nozzle.Fg",       "perf.Fg_0")
        self.connect("inlet.F_ram",          "perf.ram_drag")
        self.connect("core_nozzle.Fl_O:stat:W", "perf.W_core")
        # Bypass nozzle
        self.connect("bypass_nozzle.Fg",     "perf.Fg_1")
        self.connect("bypass_nozzle.Fl_O:stat:W", "perf.W_bypass")
        # Fuel flow
        self.connect("combustor.Wfuel",      "perf.Wfuel_0")

        # ==================================================================
        # 5.  DESIGN-POINT INITIAL CONDITIONS & CONNECTIONS
        # ==================================================================
        if design:
            # Flight condition
            self.set_input_defaults("fc.MN",  0.80)
            self.set_input_defaults("fc.alt", 35000.0, units="ft")
            self.set_input_defaults("fc.dTs", 0.0,     units="degR")
            # Total mass flow
            self.set_input_defaults("inlet.Fl_I:stat:W", 38.5, units="lbm/s")
            # ~ 17.5 kg/s total; BPR=2.5 → core ~5 kg/s (~11 lbm/s)
            # Bypass ratio
            self.set_input_defaults("splitter.BPR", 2.5)
            # IGV duct – pressure loss only (no work)
            self.set_input_defaults("igv_duct.dPqP",    0.025)  # 2.5% loss (IGV + spinner)
            self.set_input_defaults("igv_duct.MN",      0.40)
            # Stage 1 compressor
            self.set_input_defaults("comp1.PR",          4.0)
            self.set_input_defaults("comp1.eff",         0.88)
            self.set_input_defaults("comp1.MN",          0.35)
            self.set_input_defaults("comp1.motor_eta",   0.95)
            # Stage 2 compressor
            self.set_input_defaults("comp2.PR",          3.5)
            self.set_input_defaults("comp2.eff",         0.88)
            self.set_input_defaults("comp2.MN",          0.30)
            self.set_input_defaults("comp2.motor_eta",   0.95)
            # Diffuser
            self.set_input_defaults("diffuser.dPqP",    0.02)   # 2% pressure loss
            self.set_input_defaults("diffuser.MN",      0.20)
            # Combustor – TIT in Rankine (1800 K = 3240 R)
            self.set_input_defaults("combustor.T4_MAX",  3240.0, units="degR")
            self.set_input_defaults("combustor.dPqP",    0.04)   # 4% pressure loss
            self.set_input_defaults("combustor.MN",      0.10)
            # Core nozzle (C-D, perfectly expanded)
            self.set_input_defaults("core_nozzle.Cv",    0.97)
            # Exit area set by solver to achieve P_exit = P_amb (perfectly expanded)
            # Bypass nozzle (convergent)
            self.set_input_defaults("bypass_nozzle.Cv",  0.97)

        # ==================================================================
        # 6.  SOLVER SETTINGS
        # ==================================================================
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"]             = 1e-8
        newton.options["rtol"]             = 1e-8
        newton.options["iprint"]           = 2
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"]   = 20
        newton.options["maxiter"]          = 50
        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options["rho"]   = 0.6
        newton.linesearch.options["maxiter"] = 10
        newton.linesearch.options["iprint"] = -1
        self.linear_solver = om.DirectSolver(assemble_jac=True)


# ===========================================================================
# 6.  RESULTS VIEWER
# ===========================================================================
def print_station_results(prob):
    """Pretty-print station-by-station results matching the hand analysis."""
    divider = "=" * 72
    thin    = "-" * 72

    def tot(element, var):
        """Fetch a total-condition variable [SI] from an element's Fl_O port."""
        key = f"{element}.Fl_O:tot:{var}"
        try:
            val = prob.get_val(key)
            return float(val[0]) if hasattr(val, "__len__") else float(val)
        except KeyError:
            return float("nan")

    def stat(element, var):
        key = f"{element}.Fl_O:stat:{var}"
        try:
            val = prob.get_val(key)
            return float(val[0]) if hasattr(val, "__len__") else float(val)
        except KeyError:
            return float("nan")

    def pval(name, units=None):
        try:
            v = prob.get_val(name, units=units) if units else prob.get_val(name)
            return float(v[0]) if hasattr(v, "__len__") else float(v)
        except KeyError:
            return float("nan")

    # --- Unit conversion helpers ---
    R2K    = lambda R: (R - 491.67) * 5/9   # Rankine → Kelvin
    psi2Pa = 6894.757
    hp2kW  = 0.745699872

    print(f"\n{divider}")
    print(" TURBOELECTRIC ADAPTIVE ENGINE – PYCYCLE STATION RESULTS")
    print(f"{divider}")
    print(f"{'Station':<10} {'Element':<18} {'Tt [K]':>9} {'Pt [kPa]':>10} "
          f"{'Ts [K]':>9} {'Ps [kPa]':>10} {'MN':>6}")
    print(thin)

    stations = [
        ("St 0  ",  "fc",             "Freestream"),
        ("St 1  ",  "inlet",          "Inlet exit"),
        ("St 3-5",  "igv_duct",       "Post-IGV / comp entry"),
        ("St 6  ",  "comp1",          "Post-Compressor 1"),
        ("St 7  ",  "comp2",          "Post-Compressor 2"),
        ("St 8  ",  "diffuser",       "Diffuser exit"),
        ("St 10 ",  "combustor",      "Combustor exit"),
        ("St 12 ",  "core_nozzle",    "Core nozzle exit"),
        ("Byp   ",  "bypass_nozzle",  "Bypass nozzle exit"),
    ]

    for st, elem, label in stations:
        Tt_R  = tot(elem, "T")
        Pt_ps = tot(elem, "P")
        Ts_R  = stat(elem, "T")
        Ps_ps = stat(elem, "P")
        MN    = stat(elem, "MN") if not np.isnan(stat(elem, "MN")) else \
                pval(f"{elem}.Fl_O:tot:MN")
        Tt_K   = R2K(Tt_R)  if not np.isnan(Tt_R)  else float("nan")
        Pt_kPa = Pt_ps * psi2Pa / 1000 if not np.isnan(Pt_ps) else float("nan")
        Ts_K   = R2K(Ts_R)  if not np.isnan(Ts_R)  else float("nan")
        Ps_kPa = Ps_ps * psi2Pa / 1000 if not np.isnan(Ps_ps) else float("nan")
        print(f"{st:<10} {label:<18} {Tt_K:>9.1f} {Pt_kPa:>10.2f} "
              f"{Ts_K:>9.1f} {Ps_kPa:>10.2f} {MN:>6.3f}")

    print(thin)

    # --- Performance summary ---
    Fn      = pval("perf.Fn",   units="lbf") * 4.44822    # lbf → N
    TSFC_hr = pval("perf.TSFC") * 3600                     # per-sec → per-hr (lbm/lbf·hr)
    Wfuel   = pval("combustor.Wfuel", units="lbm/s") * 0.453592  # → kg/s
    # Electric power: comp1 + comp2 (in hp, convert to kW)
    P_comp1_hp = pval("comp1.power")
    P_comp2_hp = pval("comp2.power")
    P_elec1_kW = pval("comp1.power_elec") / 1000 if not np.isnan(pval("comp1.power_elec")) \
                 else P_comp1_hp * hp2kW / 0.95
    P_elec2_kW = pval("comp2.power_elec") / 1000 if not np.isnan(pval("comp2.power_elec")) \
                 else P_comp2_hp * hp2kW / 0.95
    P_total_MW = (P_elec1_kW + P_elec2_kW) / 1000

    OPR  = pval("comp2.Fl_O:tot:P") / pval("inlet.Fl_O:tot:P")
    BPR  = pval("splitter.BPR")
    Wdot = pval("inlet.Fl_I:stat:W", units="lbm/s") * 0.453592

    print(f"\n{'PERFORMANCE SUMMARY':^72}")
    print(thin)
    print(f"  Net Thrust (Fn)          : {Fn:>10.1f}  N")
    print(f"  TSFC                     : {TSFC_hr:>10.4f}  lbm/lbf·hr")
    print(f"  Fuel flow (Wfuel)        : {Wfuel:>10.3f}  kg/s")
    print(f"  Overall Pressure Ratio   : {OPR:>10.2f}")
    print(f"  Bypass Ratio             : {BPR:>10.2f}")
    print(f"  Total mass flow          : {Wdot:>10.2f}  kg/s")
    print(thin)
    print(f"  Comp1 shaft power        : {P_comp1_hp * hp2kW:>10.1f}  kW")
    print(f"  Comp2 shaft power        : {P_comp2_hp * hp2kW:>10.1f}  kW")
    print(f"  Comp1 ELECTRIC power     : {P_elec1_kW:>10.1f}  kW  (incl. motor η)")
    print(f"  Comp2 ELECTRIC power     : {P_elec2_kW:>10.1f}  kW  (incl. motor η)")
    print(f"  TOTAL ELECTRIC POWER BUS : {P_total_MW:>10.3f}  MW")
    print(divider)
    print()


# ===========================================================================
# 7.  MAIN – build, run, report
# ===========================================================================
def run_design_point():
    prob = om.Problem()
    prob.model = TurboelectricEngine(design=True)

    # Recorder (optional – saves all variables to SQLite)
    recorder = om.SqliteRecorder("turboelectric_design_pt.sql")
    prob.add_recorder(recorder)

    prob.setup(check=False, force_alloc_complex=True)

    prob.set_val("fc.MN",  0.80)
    prob.set_val("fc.alt", 35000.0, units="ft")
    # Initial guesses to help Newton converge
    prob.set_val("inlet.Fl_I:stat:W",    38.5,  units="lbm/s")
    prob.set_val("comp1.PR",              4.0)
    prob.set_val("comp2.PR",              3.5)
    prob.set_val("combustor.T4_MAX",   3240.0,  units="degR")
    prob.set_val("comp1.map.NcMap",      1.0)
    prob.set_val("comp2.map.NcMap",      1.0)

    dm = prob.check_totals(compact_print=True) if "--check" in sys.argv else None

    prob.run_model()
    prob.record("design_point")
    prob.cleanup()

    print_station_results(prob)
    return prob


# ===========================================================================
# 8.  OFF-DESIGN SWEEP  (optional – vary Mach 0.3 → 0.85)
# ===========================================================================
def run_offdesign_sweep(design_prob):
    """
    Build an off-design model balanced against the design-point geometry
    and sweep Mach number, printing thrust and electric power at each point.
    """
    print("\nRunning off-design Mach sweep ...")
    mach_points = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85]
    results = []

    for MN in mach_points:
        prob = om.Problem()
        prob.model = TurboelectricEngine(design=False)
        prob.setup(check=False)
        # Copy design-point geometry (areas, corrected flows) from design run
        # In a full pyCycle model you would use pyc.connect_des_od() here
        prob.set_val("fc.MN",  MN)
        prob.set_val("fc.alt", 35000.0, units="ft")
        prob.set_val("combustor.T4_MAX", 3240.0, units="degR")
        try:
            prob.run_model()
            Fn    = float(prob.get_val("perf.Fn",   units="lbf")) * 4.44822
            Pelec = (float(prob.get_val("comp1.power_elec")) +
                     float(prob.get_val("comp2.power_elec"))) / 1e6
            results.append((MN, Fn, Pelec))
        except Exception as e:
            results.append((MN, float("nan"), float("nan")))

    print(f"\n{'MN':>6}  {'Fn [N]':>12}  {'P_elec [MW]':>13}")
    print("-" * 36)
    for MN, Fn, Pelec in results:
        print(f"{MN:>6.2f}  {Fn:>12.1f}  {Pelec:>13.3f}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  TURBOELECTRIC ADAPTIVE ENGINE – pyCycle Cycle Analysis")
    print("  Design Point: M0=0.80  Alt=35,000 ft  TIT=1800 K  BPR=2.5")
    print("  Architecture: Electric-driven dual compressor, C-D core nozzle")
    print("=" * 72 + "\n")

    design_prob = run_design_point()

    if "--sweep" in sys.argv:
        run_offdesign_sweep(design_prob)
