from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

import math
import numpy as np
from gemseo.core.discipline import Discipline


class Weights(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names([
            # Wing inputs
            "s_w",          # Wing area [ft^2]
            "w_fw",         # Weight of fuel in wing [lbs]
            "ar_w",         # Wing aspect ratio
            "lambda_c4",    # Quarter-chord sweep angle [rad]
            "q",            # Dynamic pressure at cruise [psf]
            "lambda_w",     # Wing taper ratio
            "t_c",          # Thickness-to-chord ratio
            "n_z",          # Ultimate load factor
            "w_0",          # Takeoff gross weight [lbs]
            
            # HTP inputs
            "s_ht",         # Horizontal tail area [ft^2]
            "lambda_ht",    # Horizontal tail taper ratio
            
            # Fuselage inputs
            "s_fus",        # Fuselage wetted area [ft^2]
            "len_ht",       # Horizontal tail length [ft]
            "len_fs",       # Fuselage length [ft]
            "d_fs",         # Fuselage diameter [ft]
            "v_p",          # Cabin volume [ft^3]
            "delta_p",      # Cabin pressure differential [psi]
            
            # Fuel system inputs
            "q_tot",        # Total fuel quantity [gal]
            "q_int",        # Internal fuel quantity [gal]
            "n_tank",       # Number of fuel tanks
            "n_eng",        # Number of engines
            
            # Flight control inputs
            "b_w",          # Wingspan [ft]
            
            # Avionics input
            "w_uav",        # Uninstalled avionics weight [lbs]
            
            # Vertical tail inputs
            "f_tail",       # Fraction of tail area relative to vertical tail
            "s_vt",         # Vertical tail area [ft^2]
            "lambda_vt",    # Vertical tail quarter-chord sweep angle [rad]
            "lambda_vt_taper", # Vertical tail taper ratio
            
            # Propeller inputs
            "k_prop",       # Multiplication factor
            "n_props",      # Number of propellers
            "n_blades",     # Number of blades per propeller
            "d",            # Propeller diameter [ft]
            "p_max"         # Maximum power input to propeller [hp]
        ])
        
        self.output_grammar.update_from_names([
            "wing_weight",
            "htp_weight",
            "fuselage_weight",
            "fuel_system_weight",
            "flight_control_weight",
            "avionics_weight",
            "furnishings_weight",
            "electrical_system_weight",
            "engine_weight",
            "vertical_tail_weight",
            "propeller_weight",
            "total_weight"
        ])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # Calculate individual component weights
        wing_w = calculate_wing_weight(
            input_data["s_w"], input_data["w_fw"], input_data["ar_w"],
            input_data["lambda_c4"], input_data["q"], input_data["lambda_w"],
            input_data["t_c"], input_data["n_z"], input_data["w_0"]
        )
        
        htp_w = calculate_htp_weight(
            input_data["n_z"], input_data["w_0"], input_data["q"],
            input_data["s_ht"], input_data["t_c"], input_data["lambda_c4"],
            input_data["ar_w"], input_data["lambda_ht"]
        )
        
        fuselage_w = calculate_fuselage_weight(
            input_data["s_fus"], input_data["n_z"], input_data["w_0"],
            input_data["len_ht"], input_data["len_fs"], input_data["d_fs"],
            input_data["q"], input_data["v_p"], input_data["delta_p"]
        )
        
        fuel_sys_w = calculate_fuel_system_weight(
            input_data["q_tot"], input_data["q_int"],
            input_data["n_tank"], input_data["n_eng"]
        )
        
        flight_ctrl_w = calculate_flight_control_weight(
            input_data["len_fs"], input_data["b_w"],
            input_data["n_z"], input_data["w_0"]
        )
        
        avionics_w = calculate_avionics_weight(input_data["w_uav"])
        
        furnishings_w = calculate_furnishings_weight(input_data["w_0"])
        
        electrical_w = calculate_electrical_system_weight(fuel_sys_w, avionics_w)
        
        engine_w = calculate_turboprop_engine_weight(input_data["p_max"])
        
        vtail_w = calculate_vertical_tail_weight(
            input_data["f_tail"], input_data["n_z"], input_data["w_0"],
            input_data["q"], input_data["s_vt"], input_data["t_c"],
            input_data["lambda_vt"], input_data["ar_w"],
            input_data["lambda_vt_taper"]
        )
        
        prop_w = calculate_propeller_weight(
            input_data["k_prop"], input_data["n_props"],
            input_data["n_blades"], input_data["d"],
            input_data["p_max"]
        )

        # Calculate total weight
        total_w = (wing_w + htp_w + fuselage_w + fuel_sys_w + flight_ctrl_w +
                  avionics_w + furnishings_w + electrical_w + engine_w +
                  vtail_w + prop_w)

        return {
            "wing_weight": wing_w,
            "htp_weight": htp_w,
            "fuselage_weight": fuselage_w,
            "fuel_system_weight": fuel_sys_w,
            "flight_control_weight": flight_ctrl_w,
            "avionics_weight": avionics_w,
            "furnishings_weight": furnishings_w,
            "electrical_system_weight": electrical_w,
            "engine_weight": engine_w,
            "vertical_tail_weight": vtail_w,
            "propeller_weight": prop_w,
            "total_weight": total_w
        }

# Keep all the original calculation functions


def calculate_wing_weight(s_w, w_fw, ar_w, lambda_c4, q, lambda_w, t_c, n_z, w_0):
    # s_w: Wing area (ft^2)
    # w_fw: Weight of fuel in the wing (lbs)
    # ar_w: Wing aspect ratio
    # lambda_c4: Quarter-chord sweep angle (radians)
    # q: Dynamic pressure at cruise (psf)
    # lambda_w: Wing taper ratio
    # t_c: Thickness-to-chord ratio
    # n_z: Ultimate load factor
    # w_0: Takeoff gross weight (lbs)
    wing_weight = (
        0.036 
        * s_w**0.758 
        * w_fw**0.0035 
        * (ar_w / (math.cos(lambda_c4)**2))**0.6 
        * q**0.006 
        * lambda_w**0.04 
        * ((100 * t_c) / math.cos(lambda_c4))**-0.3 
        * (n_z * w_0)**0.49
    )
    # wing_weight: Calculated wing weight (lbs)
    return wing_weight

def calculate_htp_weight(n_z, w_0, q, s_ht, t_c, lambda_c4, ar_w, lambda_ht):
    # n_z: Ultimate load factor
    # w_0: Takeoff gross weight (lbs)
    # q: Dynamic pressure at cruise (psf)
    # s_ht: Horizontal tail area (ft^2)
    # t_c: Thickness-to-chord ratio
    # lambda_c4: Quarter-chord sweep angle (radians)
    # ar_w: Wing aspect ratio
    # lambda_ht: Horizontal tail taper ratio
    htp_weight = (
        0.016 
        * (n_z * w_0)**0.414 
        * q**0.168 
        * s_ht**0.896 
        * ((100 * t_c) / math.cos(lambda_c4))**-0.12 
        * (ar_w / (math.cos(lambda_c4)**2))**0.043 
        * lambda_ht**-0.02
    )
    # htp_weight: Calculated horizontal tail plane weight (lbs)
    return htp_weight

def calculate_fuselage_weight(s_fus, n_z, w_0, len_ht, len_fs, d_fs, q, v_p, delta_p):
    # s_fus: Fuselage wetted area (ft^2)
    # n_z: Ultimate load factor
    # w_0: Takeoff gross weight (lbs)
    # len_ht: Horizontal tail length (ft)
    # len_fs: Fuselage length (ft)
    # d_fs: Fuselage diameter (ft)
    # q: Dynamic pressure at cruise (psf)
    # v_p: Cabin volume (ft^3)
    # delta_p: Cabin pressure differential (psi)
    fuselage_weight = (
        0.052 
        * s_fus**1.086 
        * (n_z * w_0)**0.177 
        * len_ht**-0.051 
        * (len_fs / d_fs)**-0.072 
        * (q**0.241 + 11.9 * (v_p * delta_p)**0.271)
    )
    # fuselage_weight: Calculated fuselage weight (lbs)
    return fuselage_weight

def calculate_fuel_system_weight(q_tot, q_int, n_tank, n_eng):
    # q_tot: Total fuel quantity (gallons)
    # q_int: Internal fuel quantity (gallons)
    # n_tank: Number of fuel tanks
    # n_eng: Number of engines
    fuel_system_weight = (
        2.49 
        * q_tot**0.726 
        * (q_tot / (q_tot + q_int))**0.363 
        * n_tank**0.242 
        * n_eng**0.157
    )
    # fuel_system_weight: Calculated fuel system weight (lbs)
    return fuel_system_weight

def calculate_flight_control_weight(len_fs, b_w, n_z, w_0):
    # len_fs: Fuselage length (ft)
    # b_w: Wingspan (ft)
    # n_z: Ultimate load factor
    # w_0: Takeoff gross weight (lbs)
    flight_control_weight = (
        0.053 
        * len_fs**1.536 
        * b_w**0.371 
        * ((n_z * w_0 * 10**-4)**0.80)
    )
    # flight_control_weight: Calculated flight control system weight (lbs)
    return flight_control_weight

def calculate_avionics_weight(w_uav):
    # w_uav: Uninstalled avionics weight (lbs)
    avionics_weight = 2.11 * w_uav**0.933
    # avionics_weight: Calculated avionics system weight (lbs)
    return avionics_weight

def calculate_furnishings_weight(w_0):
    # w_0: Takeoff gross weight (lbs)
    furnishings_weight = 0.0582 * w_0 - 65
    # furnishings_weight: Calculated furnishings weight (lbs)
    return furnishings_weight

def calculate_electrical_system_weight(w_fs, w_av):
    # w_fs: Fuel system weight (lbs)
    # w_av: Avionics system weight (lbs)
    electrical_system_weight = 12.57 * (w_fs + w_av)**0.51
    # electrical_system_weight: Calculated electrical system weight (lbs)
    return electrical_system_weight

def calculate_turboprop_engine_weight(p_max):
    # p_max: Maximum engine power (hp)
    turboprop_engine_weight = 71.65 + 0.3658 * p_max
    # turboprop_engine_weight: Calculated turboprop engine weight (lbs)
    return turboprop_engine_weight

def calculate_vertical_tail_weight(f_tail, n_z, w_0, q, s_vt, t_c, lambda_vt, ar_w, lambda_vt_taper):
    # f_tail: Fraction of tail area relative to the vertical tail
    # n_z: Ultimate load factor
    # w_0: Takeoff gross weight (lbs)
    # q: Dynamic pressure at cruise (psf)
    # s_vt: Vertical tail area (ft^2)
    # t_c: Thickness-to-chord ratio
    # lambda_vt: Vertical tail quarter-chord sweep angle (radians)
    # ar_w: Wing aspect ratio
    # lambda_vt_taper: Vertical tail taper ratio
    vertical_tail_weight = (
        0.073
        * (1 + 0.2 * f_tail)
        * (n_z * w_0)**0.376
        * q**0.122
        * s_vt**0.873
        * ((100 * t_c) / math.cos(lambda_vt))**-0.49
        * (ar_w / (math.cos(lambda_vt)**2))**0.357
        * lambda_vt_taper**0.039
    )
    # vertical_tail_weight: Calculated vertical tail weight (lbs)
    return vertical_tail_weight

def calculate_propeller_weight(k_prop, n_props, n_blades, d, p_max):
    # k_prop: Multiplication factor
    # n_props: Number of propellers
    # n_blades: Number of blades per propeller
    # d: Diameter of the propeller (ft)
    # p_max: Maximum power input to the propeller (hp)
    propeller_weight = (
        k_prop 
        * n_props 
        * n_blades**0.391 
        * ((d * p_max) / (1000 * n_props))**0.782
    )
    # propeller_weight: Calculated propeller weight (lbs)
    return propeller_weight

# Sources:
# All equations except the propeller weight are sourced from Gudmundsson: General Aviation Aircraft Design
# Propeller weight equation source is Gundlach: Design of Unmanned Aircraft Systems

if __name__ == "__main__":
    # Create instance of the discipline
    weights_discipline = Weights()

    # Create test input data with realistic values
    test_inputs = {
        # Wing parameters
        "s_w": 180.0,          # Wing area [ft^2]
        "w_fw": 500.0,         # Fuel weight in wing [lbs]
        "ar_w": 8.0,           # Wing aspect ratio
        "lambda_c4": 0.0,      # Quarter-chord sweep angle [rad]
        "q": 100.0,            # Dynamic pressure [psf]
        "lambda_w": 0.6,       # Wing taper ratio
        "t_c": 0.12,           # Thickness-to-chord ratio
        "n_z": 3.8,            # Ultimate load factor
        "w_0": 4000.0,         # Takeoff weight [lbs]
        
        # HTP parameters
        "s_ht": 40.0,          # Horizontal tail area [ft^2]
        "lambda_ht": 0.5,      # Horizontal tail taper ratio
        
        # Fuselage parameters
        "s_fus": 300.0,        # Fuselage wetted area [ft^2]
        "len_ht": 15.0,        # Horizontal tail length [ft]
        "len_fs": 30.0,        # Fuselage length [ft]
        "d_fs": 5.0,           # Fuselage diameter [ft]
        "v_p": 400.0,          # Cabin volume [ft^3]
        "delta_p": 8.0,        # Cabin pressure differential [psi]
        
        # Fuel system parameters
        "q_tot": 100.0,        # Total fuel quantity [gal]
        "q_int": 90.0,         # Internal fuel quantity [gal]
        "n_tank": 2,           # Number of fuel tanks
        "n_eng": 1,            # Number of engines
        
        # Flight control parameters
        "b_w": 38.0,           # Wingspan [ft]
        
        # Avionics parameters
        "w_uav": 100.0,        # Uninstalled avionics weight [lbs]
        
        # Vertical tail parameters
        "f_tail": 0.0,         # Tail area fraction
        "s_vt": 30.0,          # Vertical tail area [ft^2]
        "lambda_vt": 0.0,      # Vertical tail sweep angle [rad]
        "lambda_vt_taper": 0.5, # Vertical tail taper ratio
        
        # Propeller parameters
        "k_prop": 1.0,         # Propeller factor
        "n_props": 1,          # Number of propellers
        "n_blades": 3,         # Blades per propeller
        "d": 7.0,              # Propeller diameter [ft]
        "p_max": 300.0         # Maximum power [hp]
    }

    # Execute discipline
    result = weights_discipline._run(test_inputs)
    
    # Print results
    print("\nComponent Weights Breakdown:")
    print("-" * 40)
    for component, weight in result.items():
        if component != "total_weight":
            print(f"{component:25s}: {weight:8.1f} lbs")
    print("-" * 40)
    print(f"{'total_weight':25s}: {result['total_weight']:8.1f} lbs")
