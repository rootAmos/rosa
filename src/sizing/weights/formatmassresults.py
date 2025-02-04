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
            'Airframe': to_kg(prob.get_val('w_airframe')[0], True),
            'Systems': to_kg(prob.get_val('w_systems')[0], True),
            'Powertrain': to_kg(prob.get_val('w_hybrid_ptrain')[0], True),
            'Payload': to_kg(prob.get_val('w_pay')[0], False),
            'Fuel': to_kg(prob.get_val('w_fuel')[0], True)
        }

        detailed_categories = {
            'Wing': to_kg(prob.get_val('w_wing')[0], False),
            'Fuselage': to_kg(prob.get_val('w_fuselage')[0], False),
            'Horizontal Tail': to_kg(prob.get_val('w_htail')[0], False),
            'Vertical Tail': to_kg(prob.get_val('w_vtail')[0], False),
            'Nacelles': to_kg(prob.get_val('hybrid_ptrain_weight.compute_nacelle_weight.w_nacelles')[0], True),  # Added nacelles
            'Landing Gear': to_kg(prob.get_val('w_lg')[0], False),
            'Flight Controls': to_kg(prob.get_val('w_fcs')[0], False),
            'Electrical': to_kg(prob.get_val('w_electrical')[0], False),
            'Avionics': to_kg(prob.get_val('w_avionics')[0], False),
            'Furnishings': to_kg(prob.get_val('w_furnishings')[0], False),
            'Motors': to_kg(prob.get_val('hybrid_ptrain_weight.compute_elec_ptrain_weight.w_motors')[0], True),
            'Generators': to_kg(prob.get_val('hybrid_ptrain_weight.compute_elec_ptrain_weight.w_gens')[0], True),
            'Power Electronics': to_kg(prob.get_val('hybrid_ptrain_weight.compute_elec_ptrain_weight.w_pe')[0], True),
            'Cables': to_kg(prob.get_val('hybrid_ptrain_weight.compute_elec_ptrain_weight.w_cbls')[0], True),
            'Batteries': to_kg(prob.get_val('hybrid_ptrain_weight.compute_elec_ptrain_weight.w_bat')[0], True),
            'Turbines': to_kg(prob.get_val('hybrid_ptrain_weight.compute_turbine_weight.w_turbines')[0], True),
            'Fuel System': to_kg(prob.get_val('hybrid_ptrain_weight.compute_fuel_system_weight.w_fuel_system')[0], False),
            'Fans': to_kg(prob.get_val('hybrid_ptrain_weight.compute_fan_weight.w_propulsor')[0], True),
            'Propellers': to_kg(prob.get_val('hybrid_ptrain_weight.compute_prop_weight.w_propulsor')[0], True),
            'Payload': to_kg(prob.get_val('w_pay')[0], False),
            'Fuel': to_kg(prob.get_val('w_fuel')[0], True)
        }
        
        return major_categories, detailed_categories

    @staticmethod
    def plot_results(prob):
        major_categories, detailed_categories = FormatMassResults.get_weight_categories(prob)
        
        # Sort categories by value while keeping labels and values paired
        def alternate_sort(d):
            # Sort items by value
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)
            n = len(items)
            result = []
            # Alternate between largest and smallest
            for i in range((n + 1) // 2):

                result.append(items[i])  # Add from front
                if i + (n//2) < n:
                    result.append(items[-(i + 1)])  # Add from back
            # Return labels and values separately but maintaining correspondence
            labels, values = zip(*result)
            return list(labels), list(values)
        
        major_labels, major_values = alternate_sort(major_categories)
        detailed_labels, detailed_values = alternate_sort(detailed_categories)

        # First figure - Major Categories
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111)
        
        wedges1, texts1, autotexts1 = ax1.pie(major_values, 
                                             labels=["" for _ in major_values],
                                             autopct='%1.1f%%')
        
        # Add percentage and value labels manually
        total = sum(major_values)
        for i, autotext in enumerate(autotexts1):
            pct = major_values[i]/total * 100
            autotext.set_text(f'{pct:.1f}%\n({major_values[i]:.1f} kg)')
            
        ax1.set_title('Major Weight Categories')
        plt.setp(autotexts1, color='white', size=8)
        
        # Add legend to first figure
        ax1.legend(wedges1, major_labels,
                  title="Categories",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.tight_layout()

        # Second figure - Detailed Breakdown
        fig2 = plt.figure(figsize=(14, 8))
        ax2 = fig2.add_subplot(111)
        
        wedges2, texts2, autotexts2 = ax2.pie(detailed_values,
                                             labels=["" for _ in detailed_values],
                                             autopct='%1.1f%%')
        
        # Add percentage and value labels manually
        total = sum(detailed_values)
        for i, autotext in enumerate(autotexts2):
            pct = detailed_values[i]/total * 100
            autotext.set_text(f'{pct:.1f}%\n({detailed_values[i]:.1f} kg)')
            
        ax2.set_title('Detailed Weight Breakdown')
        plt.setp(autotexts2, color='white', size=8)
        
        # Add legend to second figure
        ax2.legend(wedges2, detailed_labels,
                  title="Categories",
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
