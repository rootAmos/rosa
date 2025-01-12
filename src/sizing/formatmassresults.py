import matplotlib.pyplot as plt
import numpy as np

class FormatMassResults:
    @staticmethod
    def to_kg(val, is_newtons=True):
        return val/9.806 if is_newtons else val/2.20462

    @staticmethod
    def get_weight_categories(prob):
        # Convert all weights to kg consistently
        to_kg = FormatMassResults.to_kg  # Local reference to the static method
        
        major_categories = {
            'Airframe': to_kg(prob.get_val('weights.w_airframe')[0], True),
            'Systems': to_kg(prob.get_val('weights.systems_weight.w_systems')[0], True),
            'Powertrain': to_kg(prob.get_val('weights.hybrid_ptrain_weight.w_hybrid_ptrain')[0], True),
            'Payload': to_kg(prob.get_val('w_payload')[0], False)
        }

        detailed_categories = {
            'Wing': to_kg(prob.get_val('weights.wing_weight.w_wing')[0], False),
            'Fuselage': to_kg(prob.get_val('weights.fuselage_weight.w_fuselage')[0], False),
            'Horizontal Tail': to_kg(prob.get_val('weights.w_htail')[0], False),
            'Vertical Tail': to_kg(prob.get_val('weights.w_vtail')[0], False),
            'Landing Gear': to_kg(prob.get_val('weights.landing_gear_weight.w_lg')[0], False),
            'Flight Controls': to_kg(prob.get_val('weights.w_fcs')[0], True),
            'Electrical': to_kg(prob.get_val('weights.w_electrical')[0], True),
            'Avionics': to_kg(prob.get_val('weights.w_avionics')[0], True),
            'Furnishings': to_kg(prob.get_val('weights.w_furnishings')[0], False),
            'Motors': to_kg(prob.get_val('weights.hybrid_ptrain_weight.elec_powertrain.w_motors')[0], True),
            'Generators': to_kg(prob.get_val('weights.hybrid_ptrain_weight.elec_powertrain.w_gens')[0], True),
            'Power Electronics': to_kg(prob.get_val('weights.hybrid_ptrain_weight.elec_powertrain.w_pe')[0], True),
            'Cables': to_kg(prob.get_val('weights.hybrid_ptrain_weight.elec_powertrain.w_cbls')[0], True),
            'Batteries': to_kg(prob.get_val('weights.hybrid_ptrain_weight.elec_powertrain.w_bat')[0], True),
            'Turbines': to_kg(prob.get_val('weights.hybrid_ptrain_weight.turbine_engine.w_turbines')[0], True),
            'Fuel System': to_kg(prob.get_val('weights.hybrid_ptrain_weight.fuel_system.w_fuel_system')[0], False),
            'Propellers': to_kg(prob.get_val('weights.hybrid_ptrain_weight.propeller.w_props')[0], True),
            'Payload': to_kg(prob.get_val('w_payload')[0], False)
        }
        
        return major_categories, detailed_categories

    @staticmethod
    def plot_results(prob):
        major_categories, detailed_categories = FormatMassResults.get_weight_categories(prob)
        total_weight = sum(detailed_categories.values())

        # MATLAB default colors
        matlab_colors = [
            '#0072BD',  # Dark blue
            '#D95319',  # Dark orange
            '#EDB120',  # Dark yellow
            '#7E2F8E',  # Dark purple
            '#77AC30',  # Medium green
            '#4DBEEE',  # Light blue
            '#A2142F'   # Dark red
        ]
        
        # Extend the color list if we need more colors
        detailed_colors = matlab_colors * 3  # Multiply list to ensure enough colors

        # First figure - Major Categories
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        wedges1, texts1 = ax1.pie(major_categories.values(), 
                                 labels=None,
                                 colors=matlab_colors[:len(major_categories)])
        ax1.set_title('Major Weight Categories')
        
        # Create legend labels with mass and percentages for major categories
        major_legend_labels = [f"{component} ({value:.1f} kg, {value/total_weight*100:.1f}%)"
                             for component, value in major_categories.items()]
        
        ax1.legend(wedges1, major_legend_labels,
                  title="Categories",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()

        # Second figure - Detailed Breakdown
        fig2 = plt.figure(figsize=(12, 8))
        ax2 = fig2.add_subplot(111)
        wedges2, texts2 = ax2.pie(detailed_categories.values(),
                                 labels=None,
                                 colors=detailed_colors[:len(detailed_categories)])
        ax2.set_title('Detailed Weight Breakdown')
        
        # Create legend labels with mass and percentages for detailed categories
        legend_labels = [f"{component} ({value:.1f} kg, {value/total_weight*100:.1f}%)"
                        for component, value in detailed_categories.items()]
        
        ax2.legend(wedges2, legend_labels,
                  title="Components",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()

        plt.show()

    @staticmethod
    def print_tables(prob):
        major_categories, detailed_categories = FormatMassResults.get_weight_categories(prob)
        total_weight = sum(detailed_categories.values())

        # Print major categories table
        print("\nMajor Categories Mass Breakdown:")
        print("-" * 45)
        print(f"{'Category':<25} {'Mass (kg)':>12} {'Percentage':>8}")
        print("-" * 45)
        for category, mass in major_categories.items():
            percentage = (mass / total_weight) * 100
            print(f"{category:<25} {mass:>12.1f} {percentage:>7.1f}%")
        print("-" * 45)
        print(f"{'Total':<25} {total_weight:>12.1f} {100:>7.1f}%")

        # Print detailed categories table
        print("\nDetailed Components Mass Breakdown:")
        print("-" * 45)
        print(f"{'Component':<25} {'Mass (kg)':>12} {'Percentage':>8}")
        print("-" * 45)
        for component, mass in detailed_categories.items():
            percentage = (mass / total_weight) * 100
            print(f"{component:<25} {mass:>12.1f} {percentage:>7.1f}%")
        print("-" * 45)
        print(f"{'Total':<25} {total_weight:>12.1f} {100:>7.1f}%")

