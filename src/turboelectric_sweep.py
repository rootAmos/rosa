"""
Turboelectric Adaptive Engine – Mach Sweep
==========================================
Standalone thermodynamic cycle analysis using the same station
parameters as turboelectric_engine.py.  Runs without pyCycle;
produces a six-panel performance plot.

Design parameters (fixed):
  Alt  = 35 000 ft
  BPR  = 2.5
  PR1  = 4.0  (stage-1 electric compressor)
  PR2  = 3.5  (stage-2 electric compressor)
  TIT  = 1 800 K  (combustor-exit total temperature)
  η_c  = 0.88    (isentropic efficiency, both compressor stages)
  η_b  = 0.995   (combustor efficiency)
  η_m  = 0.95    (motor + inverter efficiency)
  Cv   = 0.97    (both nozzles)

Usage:
  python src/turboelectric_sweep.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – saves PNG
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GAMMA_C = 1.40        # cold air
GAMMA_H = 1.33        # hot combustion products
CP_C    = 1005.0      # J/(kg·K)
CP_H    = 1150.0      # J/(kg·K)
R_AIR   = 287.058     # J/(kg·K)
LHV     = 43.2e6      # J/kg  Jet-A lower heating value

# ---------------------------------------------------------------------------
# Design parameters
# ---------------------------------------------------------------------------
ALT_FT   = 35_000.0
W_TOTAL  = 38.5 * 0.453592   # kg/s  (38.5 lbm/s)
BPR      = 2.5
PR1      = 4.0
PR2      = 3.5
ETA_C    = 0.88
ETA_B    = 0.995
ETA_M    = 0.95
CV_CORE  = 0.97
CV_BYP   = 0.97
T4_MAX   = 1800.0             # K
DPQP_IGV  = 0.025
DPQP_DIFF = 0.02
DPQP_COMB = 0.04


# ===========================================================================
# ISA atmosphere
# ===========================================================================
def isa(alt_ft: float):
    """Return (T [K], P [Pa], a [m/s]) at altitude alt_ft using ISA."""
    alt_m = alt_ft * 0.3048
    T_sl, P_sl = 288.15, 101_325.0
    g = 9.80665
    if alt_m <= 11_000.0:
        lapse = -0.0065          # K/m  (troposphere)
        T = T_sl + lapse * alt_m
        P = P_sl * (T / T_sl) ** (-g / (R_AIR * lapse))
    else:                        # stratosphere (isothermal to 20 km)
        T_11 = T_sl - 0.0065 * 11_000.0   # 216.65 K
        P_11 = P_sl * (T_11 / T_sl) ** (-g / (R_AIR * (-0.0065)))
        T = T_11
        P = P_11 * np.exp(-g * (alt_m - 11_000.0) / (R_AIR * T_11))
    a = np.sqrt(GAMMA_C * R_AIR * T)
    return T, P, a


# ===========================================================================
# Engine cycle
# ===========================================================================
def engine_cycle(MN: float) -> dict:
    """
    Run the turboelectric cycle at flight Mach number MN.
    Returns a dict of performance quantities.
    """
    T0, P0, a0 = isa(ALT_FT)
    V0 = MN * a0                                          # freestream velocity

    # --- Station 0: freestream total ---
    Tt0 = T0 * (1.0 + (GAMMA_C - 1.0) / 2.0 * MN**2)
    Pt0 = P0 * (Tt0 / T0) ** (GAMMA_C / (GAMMA_C - 1.0))

    # --- Station 1: inlet exit (ram recovery ≈ 1 subsonic) ---
    Pt1 = Pt0
    Tt1 = Tt0

    # --- Mass-flow split ---
    W_core = W_TOTAL / (1.0 + BPR)    # kg/s
    W_byp  = W_TOTAL - W_core

    # --- Station 5: IGV duct exit ---
    Pt5 = Pt1 * (1.0 - DPQP_IGV)
    Tt5 = Tt1

    # --- Station 6: post-compressor-1 ---
    tau1 = PR1 ** ((GAMMA_C - 1.0) / GAMMA_C)
    Tt6  = Tt5 * (1.0 + (tau1 - 1.0) / ETA_C)
    Pt6  = Pt5 * PR1

    # --- Station 7: post-compressor-2 ---
    tau2 = PR2 ** ((GAMMA_C - 1.0) / GAMMA_C)
    Tt7  = Tt6 * (1.0 + (tau2 - 1.0) / ETA_C)
    Pt7  = Pt6 * PR2

    # --- Station 8: diffuser exit ---
    Pt8 = Pt7 * (1.0 - DPQP_DIFF)
    Tt8 = Tt7

    # --- Station 10: combustor exit ---
    Tt10 = T4_MAX
    # Energy balance: W_core·Cp_c·Tt8 + Wf·LHV·η_b = (W_core+Wf)·Cp_h·Tt10
    Wf = W_core * (CP_H * Tt10 - CP_C * Tt8) / (LHV * ETA_B - CP_H * Tt10)
    Pt10 = Pt8 * (1.0 - DPQP_COMB)
    W_core_exit = W_core + Wf

    # --- Core nozzle (CD, fully expanded to P0) ---
    exp_core = (P0 / Pt10) ** ((GAMMA_H - 1.0) / GAMMA_H)
    Vjet_core = CV_CORE * np.sqrt(max(0.0, 2.0 * CP_H * Tt10 * (1.0 - exp_core)))

    # --- Bypass nozzle (CV, check choke) ---
    PR_crit = ((GAMMA_C + 1.0) / 2.0) ** (GAMMA_C / (GAMMA_C - 1.0))  # ≈ 1.893
    if Pt1 / P0 >= PR_crit:
        # Choked
        T_exit_byp = Tt1 * 2.0 / (GAMMA_C + 1.0)
        Vjet_byp   = CV_BYP * np.sqrt(GAMMA_C * R_AIR * T_exit_byp)
    else:
        exp_byp  = (P0 / Pt1) ** ((GAMMA_C - 1.0) / GAMMA_C)
        Vjet_byp = CV_BYP * np.sqrt(max(0.0, 2.0 * CP_C * Tt1 * (1.0 - exp_byp)))

    # --- Net thrust ---
    Fg_core = W_core_exit * Vjet_core
    Fg_byp  = W_byp * Vjet_byp
    F_ram   = W_TOTAL * V0
    Fn = Fg_core + Fg_byp - F_ram         # N

    # --- TSFC [lbm/(lbf·hr)] ---
    Wf_lbh  = Wf * 2.20462 * 3600.0
    Fn_lbf  = Fn / 4.44822
    TSFC    = Wf_lbh / Fn_lbf if Fn_lbf > 0 else float("nan")

    # --- Electric power [MW] ---
    Pshaft1 = W_core * CP_C * (Tt6 - Tt5)
    Pshaft2 = W_core * CP_C * (Tt7 - Tt6)
    P_elec  = (Pshaft1 + Pshaft2) / ETA_M / 1e6   # MW

    # --- Overall pressure ratio ---
    OPR = Pt10 / Pt1

    # --- Specific thrust [N·s/kg] ---
    Isp_spec = Fn / W_TOTAL

    return dict(
        Fn=Fn,
        TSFC=TSFC,
        Wf=Wf,
        P_elec_MW=P_elec,
        Pshaft1_kW=Pshaft1 / 1e3,
        Pshaft2_kW=Pshaft2 / 1e3,
        OPR=OPR,
        Isp_spec=Isp_spec,
        Vjet_core=Vjet_core,
        Vjet_byp=Vjet_byp,
        V0=V0,
        Tt0=Tt0, Tt5=Tt5, Tt6=Tt6, Tt7=Tt7,
        Tt8=Tt8, Tt10=Tt10,
        Pt1=Pt1, Pt10=Pt10, P0=P0,
    )


# ===========================================================================
# Mach sweep
# ===========================================================================
def run_sweep(mach_range=None):
    if mach_range is None:
        mach_range = np.linspace(0.30, 0.85, 45)

    rows = [engine_cycle(MN) for MN in mach_range]

    def col(key):
        return np.array([r[key] for r in rows])

    return mach_range, {
        "Fn":        col("Fn"),
        "TSFC":      col("TSFC"),
        "Wf":        col("Wf"),
        "P_elec":    col("P_elec_MW"),
        "OPR":       col("OPR"),
        "Isp_spec":  col("Isp_spec"),
        "Vjet_core": col("Vjet_core"),
        "Vjet_byp":  col("Vjet_byp"),
        "V0":        col("V0"),
        "Tt5":       col("Tt5"),
        "Tt6":       col("Tt6"),
        "Tt7":       col("Tt7"),
        "Tt8":       col("Tt8"),
        "Pshaft1":   col("Pshaft1_kW"),
        "Pshaft2":   col("Pshaft2_kW"),
    }


# ===========================================================================
# Plotting
# ===========================================================================
def make_plot(MN, data, out_path="turboelectric_mach_sweep.png"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Turboelectric Adaptive Engine – Mach Sweep\n"
        f"Alt = {ALT_FT/1000:.0f} kft  |  BPR = {BPR}  |  "
        f"PR₁×PR₂ = {PR1}×{PR2}  |  TIT = {T4_MAX:.0f} K  |  "
        f"η_c = {ETA_C}  η_m = {ETA_M}",
        fontsize=12, fontweight="bold", y=1.01
    )

    style = dict(linewidth=2.0, color="#1f77b4")

    # ---- Panel 1: Net Thrust ----
    ax = axes[0, 0]
    ax.plot(MN, data["Fn"] / 1e3, **style)
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Net Thrust  [kN]")
    ax.set_title("Net Thrust")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MN[0], MN[-1])

    # ---- Panel 2: TSFC ----
    ax = axes[0, 1]
    ax.plot(MN, data["TSFC"], color="#d62728", linewidth=2.0)
    ax.set_xlabel("Mach number")
    ax.set_ylabel("TSFC  [lbm / (lbf · hr)]")
    ax.set_title("Thrust-Specific Fuel Consumption")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MN[0], MN[-1])

    # ---- Panel 3: Electric power ----
    ax = axes[0, 2]
    ax.plot(MN, data["P_elec"], color="#2ca02c", linewidth=2.0, label="Total bus")
    ax.plot(MN, data["Pshaft1"] / ETA_M / 1e3, "--", color="#98df8a",
            linewidth=1.5, label="Stage 1")
    ax.plot(MN, data["Pshaft2"] / ETA_M / 1e3, ":",  color="#006d2c",
            linewidth=1.5, label="Stage 2")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Electric Power  [MW]")
    ax.set_title("Electric Power Demand")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MN[0], MN[-1])

    # ---- Panel 4: Specific Thrust ----
    ax = axes[1, 0]
    ax.plot(MN, data["Isp_spec"], color="#9467bd", linewidth=2.0)
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Specific Thrust  [N · s / kg]")
    ax.set_title("Specific Thrust (based on total W)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MN[0], MN[-1])

    # ---- Panel 5: Jet velocity ratio ----
    ax = axes[1, 1]
    ax.plot(MN, data["Vjet_core"] / data["V0"], color="#8c564b",
            linewidth=2.0, label="Core nozzle  $V_j/V_0$")
    ax.plot(MN, data["Vjet_byp"]  / data["V0"], "--", color="#c49c94",
            linewidth=1.8, label="Bypass nozzle  $V_j/V_0$")
    ax.axhline(1.0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Jet velocity ratio  $V_j / V_0$")
    ax.set_title("Nozzle Jet Velocity Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MN[0], MN[-1])

    # ---- Panel 6: Station total temperatures ----
    ax = axes[1, 2]
    ax.plot(MN, data["Tt5"],  label="St 5  (comp entry)",     linewidth=1.8)
    ax.plot(MN, data["Tt6"],  label="St 6  (post-comp 1)",    linewidth=1.8)
    ax.plot(MN, data["Tt7"],  label="St 7  (post-comp 2)",    linewidth=1.8)
    ax.plot(MN, data["Tt8"],  label="St 8  (diffuser exit)",  linewidth=1.8)
    ax.axhline(T4_MAX, color="red", linewidth=1.2,
               linestyle="--", label=f"TIT = {T4_MAX:.0f} K")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("Total Temperature  [K]")
    ax.set_title("Station Stagnation Temperatures")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MN[0], MN[-1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    return out_path


# ===========================================================================
# Console summary table
# ===========================================================================
def print_table(MN_arr, data):
    hdr = (f"{'MN':>5}  {'Fn [kN]':>9}  {'TSFC':>7}  "
           f"{'Wf [kg/s]':>10}  {'P_elec [MW]':>12}  "
           f"{'OPR':>6}  {'Isp_sp [N·s/kg]':>17}")
    print("\n" + "=" * len(hdr))
    print("  TURBOELECTRIC MACH SWEEP  –  Alt 35 000 ft")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, MN in enumerate(MN_arr):
        print(
            f"{MN:>5.2f}  "
            f"{data['Fn'][i]/1e3:>9.2f}  "
            f"{data['TSFC'][i]:>7.4f}  "
            f"{data['Wf'][i]:>10.4f}  "
            f"{data['P_elec'][i]:>12.3f}  "
            f"{data['OPR'][i]:>6.1f}  "
            f"{data['Isp_spec'][i]:>17.1f}"
        )
    print("=" * len(hdr))
    print()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  TURBOELECTRIC ENGINE – Mach Sweep  (analytical cycle model)")
    print(f"  Alt={ALT_FT:.0f} ft  BPR={BPR}  PR1={PR1}  PR2={PR2}  "
          f"TIT={T4_MAX:.0f} K  η_c={ETA_C}")
    print("=" * 72)

    sweep_MN = np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55,
                          0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    _, data_coarse = run_sweep(sweep_MN)
    print_table(sweep_MN, data_coarse)

    # Fine resolution for smooth plot
    MN_fine, data_fine = run_sweep(np.linspace(0.30, 0.85, 200))
    make_plot(MN_fine, data_fine, out_path="turboelectric_mach_sweep.png")
