
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pytz 
from datetime import datetime, timedelta 

import sys, os
## copy of main script to test out differences. 
# Path environment
current_dir = os.getcwd()
print(current_dir)
data_path = os.path.abspath(os.path.join(current_dir, ".", "Data"))
sys.path.insert(0, data_path)

# Define manual start time in local PDT time ################# CHANGE THIS TO MATCH THE START TIME YOU DESIRE #################
pdt_tz = pytz.timezone('America/Los_Angeles')
manual_start_time_naive = datetime(2025, 6, 21, 10, 10, 0)

# Load simulation data 
pred_file_name = 'padre-sim-converted.csv' #Lat long time values from ES
pred_file_path = os.path.abspath(os.path.join(data_path, pred_file_name))
df = pd.read_csv(pred_file_path)
mask_file_name = 'spenvis_binary_mask.csv' #SPENVIS mask file only masks SAA
mask_file_path = os.path.abspath(os.path.join(data_path, mask_file_name))
mask = pd.read_csv(mask_file_path).iloc[:, 2:]


lat_resolution = 1 
lon_resolution = 1 

df['lat_i'] = (df['latitude']/lat_resolution + 90 - 1).astype(int)
df['lon_i'] = (df['longitude']/lon_resolution + 180 - 1).astype(int)

# operational mask 
df['SAA'] = np.abs(mask.values[df['lat_i'].values, df['lon_i'].values]) 
df['SAA'] = np.abs(df['SAA'].astype(bool).astype(int)-1)

df['eclipse'] = ~df['sunlit']




# Convert timestamp (assumed in seconds) to relative time and then to absolute time
df['relative_time'] = df['timestamp'] - df['timestamp'][0]
df['absolute_pdt'] = df['relative_time'].apply(lambda x: manual_start_time_naive + timedelta(seconds=x))

# --- Create Mode Columns ---

# Spacecraft Mode: Always "PAYLOAD"
df['Spacecraft_mode'] = 'PAYLOAD'

# SHARP Instrument Mode:
#   - If sunlit is False (eclipse): "IDLE"
#   - If sunlit is True & inside a restricted region (inside_restricted_region == 0): "ENGINEERING"
#   - Otherwise (sunlit True & allowed region): "SCIENCE"
df['SHARP_mode'] = 'SCIENCE'  # default to SCIENCE
df.loc[df['sunlit'] == False, 'SHARP_mode'] = 'IDLE'
df.loc[(df['sunlit'] == True) & (df['SAA']), 'SHARP_mode'] = 'ENGINEERING'

# MeDDEA Instrument Mode:
#   - If sunlit is False: "IDLE"
#   - Otherwise: "SCIENCE"
df['MeDDEA_mode'] = 'SCIENCE'
df.loc[df['sunlit'] == False, 'MeDDEA_mode'] = 'SCIENCE'
df.loc[(df['sunlit'] == True) & (df['SAA'] ), 'MeDDEA_mode'] = 'IDLE'


# --- Detect Mode Transitions ---

# Create a composite mode state to capture all three modes in one string:
df['mode_state'] = (df['Spacecraft_mode'] + "_" + 
                    df['SHARP_mode'] + "_" + 
                    df['MeDDEA_mode'])

# Identify transitions where the mode state changes from the previous row:
df['mode_transition'] = df['mode_state'] != df['mode_state'].shift()

# Extract the rows where a transition occurs:
transitions = df[df['mode_transition']].copy()

# Select relevant columns to export:
transitions_export = transitions[['absolute_pdt', 'eclipse', 'SAA', 'Spacecraft_mode', 'SHARP_mode', 'MeDDEA_mode',]]

# Export the transitions to a CSV file
output_path = os.path.abspath(os.path.join(current_dir, ".", "Output"))
save_file_name = f'mode_transitions_start_{manual_start_time_naive.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
output_csv = os.path.abspath(os.path.join(output_path, save_file_name))
transitions_export.to_csv(output_csv, index=False)
print(f"Mode transitions saved to {output_csv}")




