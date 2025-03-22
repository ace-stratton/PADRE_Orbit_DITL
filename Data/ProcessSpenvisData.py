import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata
from scipy.ndimage import label


# Path environment
path_GD = '/Users/chris/GoogleDrive/'
path2data = path_GD + 'Padre/PADRE/10_Systems Integration and Test/DITL/Data/'

flux_file_path  = path2data + 'spenvis_tpo.htm'
pos_file_path   = path2data + 'spenvis_pos.htm'


# Function to extract numerical data from SPENVIS files
def extract_numerical_data(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        if line.strip() and not line.startswith("'"):  # Ignore metadata lines
            values = line.strip().split(",")
            try:
                data.append([float(v) for v in values])
            except ValueError:
                continue  # Skip lines that are not numerical
    return pd.DataFrame(data)

# Function to read and process SPENVIS flux and position data
def read_spenvis_data(flux_file, pos_file):
    # Read data
    flux_df = extract_numerical_data(flux_file)
    pos_df = extract_numerical_data(pos_file)
    
    # Define column names for flux data
    flux_df.columns = ["Altitude_km", "Latitude_deg", "Longitude_deg", "Local_Time_hr", "Universal_Time_hr"]
    
    # Define column names for position data
    pos_df.columns = ["B_Gauss", "L_R_E", "Flux_cm2_s"]
    
    # Merge data
    merged_df = pd.concat([flux_df, pos_df], axis=1)
    
    # Drop NaN values in Latitude, Longitude, and Flux
    merged_df = merged_df.dropna(subset=["Latitude_deg", "Longitude_deg", "Flux_cm2_s"])
    
    return merged_df

def remove_small_islands(binary_filter,min_cluster=5):
    # Convert binary_filter to a NumPy array for processing
    binary_array = binary_filter.values

    # Label connected components
    labeled_array, num_features = label(binary_array)

    # Count occurrences of each label
    unique, counts = np.unique(labeled_array, return_counts=True)

    # Ensure 0 is not included in small regions (background)
    small_regions = set(unique[(counts < min_cluster) & (unique != 0)])

    # Remove small regions by setting them to 0
    binary_cleaned = np.where(np.isin(labeled_array, small_regions), 0, binary_array)

    return binary_cleaned


def create_binary_filter(flux_grid, flux_threshold=10, island_filter=True):
    # Remove NaNs and negative values
    flux_grid = flux_grid.fillna(0)
    flux_grid[flux_grid < 0] = 0

    # Create a binary filter
    binary_filter = (flux_grid < flux_threshold).astype(int)

    if island_filter: 
        return remove_small_islands(binary_filter)

    else: return binary_filter





# Read and process the data
merged_df = read_spenvis_data(flux_file_path, pos_file_path)

# Define interpolation grid at 1-degree resolution
lat_range = np.arange(-90, 91, 1)  # 1-degree steps
lon_range = np.arange(-180, 179, 1)
lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)

# Perform interpolation
points = merged_df[["Longitude_deg", "Latitude_deg"]].values
values = merged_df["Flux_cm2_s"].values
flux_grid = griddata(points, values, (lon_grid, lat_grid), method='linear')

# Convert grid to DataFrame
flux_grid_df = pd.DataFrame(flux_grid, index=lat_range, columns=lon_range)

# Save to CSV in matrix format
flux_grid_df.to_csv(path2data + "spenvis_flux_grid_interpolated.csv", index=True, header=True)



# Flux 


binary_cleaned = create_binary_filter(flux_grid_df)



# Convert back to DataFrame and retain latitudes/longitudes
binary_cleaned_df = pd.DataFrame(binary_cleaned, columns=lon_range)
binary_cleaned_df.insert(0, 'latitude', lat_range)



# Save to CSV in matrix format
binary_cleaned_df.to_csv(path2data + "spenvis_binary_mask.csv", index=True, header=True)


plt.figure() 
plt.contourf(binary_cleaned_df)
plt.show 


