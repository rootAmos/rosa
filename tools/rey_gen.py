import pandas as pd
import numpy as np

# Constants
density = 1.225  # kg/m^3
viscosity = 1.81e-5  # Pa.s (dynamic viscosity of dry air at 15°C)
speed_range = np.arange(10, 101, 30)  # speeds between 10 and 100 m/s in 30 m/s increments
length_range = np.arange(0.1, 2.2, 0.2)  # characteristic lengths between 0.1m and 2.1m at 20 cm increments

# Initialize an empty list to store the results
results = []

# Generate the combinations
for length in length_range:
    for speed in speed_range:
        # Calculate Reynolds number: Re = (density * speed * length) / viscosity
        reynolds_number = (density * speed * length) / viscosity
        # Calculate Mach number: Mach = speed / sound_speed
        # For standard atmospheric conditions (speed of sound at 20°C is approximately 343 m/s)
        mach_number = speed / 343
        results.append([reynolds_number, mach_number])

# Create a DataFrame
df = pd.DataFrame(results, columns=["Reynolds Number", "Mach Number"])

# Save to CSV
df.to_csv('tools/mach_reynolds_combinations.csv', index=False)

print("CSV file has been generated.")
