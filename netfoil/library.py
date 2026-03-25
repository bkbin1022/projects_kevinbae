import tensorflow as tf
import numpy as np
import subprocess
import os
import pandas as pd

def generate_xfoil_data(naca_code, alpha_start, alpha_end, alpha_step, reynolds):
    polar_filename = f"polar_{naca_code}.txt"
    
    # Delete old polar file if it exists
    if os.path.exists(polar_filename):
        os.remove(polar_filename)

    commands = [
        f"NACA {naca_code}",   # Generate the airfoil shape
        "PPAR",        # Enter Paneling Parameters
        "N 200",       # Increase number of nodes to 200 (standard is ~160)
        "P",           # Set node distribution 
        "1",           # Use a distribution that clusters more at the LE/TE
        "", "",        # Enter twice to return to main menu
        "PANE",                # Smooth the paneling (critical for convergence)
        "OPER",                # Enter operation mode
        f"Visc {reynolds}",    # Turn on viscous mode and set Re
        "ITER 200",            # Increase iterations for better convergence
        "PACC",                # Toggle Polar Accumulation on
        polar_filename,        # Name of the save file
        "",                    # Hit enter to skip dump file
        f"ASEQ {alpha_start} {alpha_end} {alpha_step}", # Run the alpha sweep
        "PACC",                # Toggle Polar Accumulation off
        "QUIT"                 # Exit XFOIL
    ]
    
    command_string = "\n".join(commands) + "\n"

    # 2. Run XFOIL silently using subprocess
    process = subprocess.Popen(
        ['xvfb-run', 'xfoil'], 
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # 2. Attempt to communicate with a 30-second limit
        stdout, stderr = process.communicate(input=command_string, timeout=30)
        
    except subprocess.TimeoutExpired:
        # 3. Clean up the process if it hangs
        process.kill() # Force kill the process
        # On Linux/Codespaces, sometimes xvfb needs an extra cleanup
        stdout, stderr = process.communicate() 
        print(f"TIMEOUT: NACA {naca_code} at Re {reynolds} took too long. Skipping...")
        return pd.DataFrame()

    # 3. polar file into pandas
    try:
        with open(polar_filename) as f:
            lines = f.readlines()
        data_start = next(i for i, l in enumerate(lines) if l.strip().startswith('---')) + 1 # data_start skips useless lines in polar_xxxx.txt
        df = pd.read_csv(polar_filename, skiprows=data_start, sep=r'\s+',                    # sep=r'\s+' to handle irregular spacing.
                         names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr', 'Top_Itr', 'Bot_Itr'])
        
        # Add inputs to the dataframe for the NN
        df['NACA'] = naca_code
        df['Re'] = reynolds

        os.remove(polar_filename)  # remove polar_####.txt

        return df[['NACA', 'Re', 'alpha', 'CL', 'CD']] # Return only what we need
        
    except FileNotFoundError:
        print(f"XFOIL failed to converge for NACA {naca_code} at Re={reynolds}")
        return pd.DataFrame() # Return empty df if it failed


def generate_naca():
    naca_list = []
    
    # 1. Symmetrical Airfoils (00XX) - Teaches thickness vs drag
    # Range: 6% (thin) to 24% (thick)
    symmetrical = [f"00{t:02d}" for t in [6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 22, 24]]
    naca_list.extend(symmetrical)
    
    # 2. Low Camber (24XX) - Standard general aviation shapes
    low_camber = [f"24{t:02d}" for t in [8, 10, 12, 14, 15, 18, 20, 24]]
    naca_list.extend(low_camber)
    
    # 3. High Camber (44XX & 64XX) - High lift/STOL shapes
    high_camber = [f"{c}4{t:02d}" for c in [4, 6] for t in [10, 12, 15, 18, 21]]
    naca_list.extend(high_camber)
    
    # 4. Camber Position Sweep (X2XX, X4XX, X6XX)
    pos_sweep = [f"4{p}{t:02d}" for p in [2, 3, 5, 6] for t in [10, 12, 15]]
    naca_list.extend(pos_sweep)
    
    # 5. Common shapes
    fine_tune = ["1408", "1410", "1412", "3412", "5412", "2312", "2512", "2612"]
    naca_list.extend(fine_tune)

    # Remove duplicates and cap at 60 (though this logic gives ~60 unique ones)
    return list(dict.fromkeys(naca_list))[:60]


# ========================= RUN XFOIL AND SAVE AS CSV ===============================
"""
if __name__ == "__main__":
    print("Running XFOIL simulation...")
    naca_pack = generate_naca()
    reynolds_list = [1e5, 5*1e5, 1e6, 5*1e6, 1e7]
    all_data_frames = []

    total = len(naca_pack) * len(reynolds_list)
    count = 0

    for code in naca_pack:
        for reynold in reynolds_list:
            count += 1
            print(f"[{count}/{total}] NACA {code} | Re {reynold:,.0f}...", end=" ", flush=True)
            
            try:
                results_df = generate_xfoil_data(naca_code=code, alpha_start=-5, alpha_end=15, alpha_step=1, reynolds=reynold)
                
                if not results_df.empty:
                    # Double check that these columns exist for the NN
                    results_df['NACA'] = code
                    results_df['Re'] = reynold
                    all_data_frames.append(results_df)
                    print("Done.")
                else:
                    print("Failed (No Convergence).")
            except Exception as e:
                print(f"Error: {e}")

    if all_data_frames:
        master_df = pd.concat(all_data_frames, ignore_index=True)
        master_df.to_csv("airfoil_dataset.csv", index=False)
        print(f"\nSuccess! Saved {len(master_df)} rows to airfoil_dataset.csv")

    # delete .bl file
    import glob
    for f in glob.glob("*.bl"):
        os.remove(f)
"""
# ===================================================================================

def create_aerospace_dataset(csv_file, batch_size=32):
    df = pd.read_csv(csv_file)
    
    # Extract physics from NACA 
    naca_str = df['NACA'].astype(str).str.zfill(4)  # ensure NACA is 4 digits
    df['camber'] = naca_str.str[0].astype(float) / 100.0
    df['camber_pos'] = naca_str.str[1].astype(float) / 10.0
    df['thickness'] = naca_str.str[2:4].astype(float) / 100.0
    

    inputs = df[['camber', 'camber_pos', 'thickness', 'Re', 'alpha']].values.astype(np.float32)
    outputs = df[['CL', 'CD']].values.astype(np.float32)
    
    # NORMALIZATION (Crucial for Neural Networks)
    # Neural networks struggle when Re is 1,000,000 but thickness is 0.12. 
    # We scale all inputs to have a mean of 0 and std of 1.
    x_mean = inputs.mean(axis=0)
    x_std = inputs.std(axis=0) + 1e-8
    inputs_scaled = (inputs - x_mean) / x_std
    
    dataset = tf.data.Dataset.from_tensor_slices((inputs_scaled, outputs))
    dataset = dataset.shuffle(buffer_size=len(inputs)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, x_mean, x_std

# --- Build the Neural Network ---
def build_surrogate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,)),   # 5 neurons in input (camber, camber_pos, thickness, Re, alpha)
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),  
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)                        # 2 neurons in output (CL, CD)
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse', 
        metrics=['mae']
    )
    
    return model


def parse_airfoil_file(uploaded_file):
    # Load coordinates, ignoring the first line (header)
    # This works for both .txt and .dat
    try:
        data = np.loadtxt(uploaded_file, skiprows=1)
    except:
        return None

    # Find the Leading Edge (point closest to x=0)
    le_idx = np.argmin(data[:, 0])
    
    # Split into Upper and Lower surfaces
    upper = data[:le_idx+1]
    lower = data[le_idx:]
    
    # Interpolate to find thickness and camber at 100 points
    x_range = np.linspace(0, 1, 100)
    y_upper = np.interp(x_range, upper[::-1, 0], upper[::-1, 1])
    y_lower = np.interp(x_range, lower[:, 0], lower[:, 1])
    
    # Calculate Geometric Properties
    thickness = y_upper - y_lower
    camber_line = (y_upper + y_lower) / 2
    
    # Extract NACA 4-digit parameters
    max_camber = np.max(camber_line) * 100
    max_camber_pos = x_range[np.argmax(camber_line)] * 10
    max_thickness = np.max(thickness) * 100
    
    return {
        "camber": round(max_camber), 
        "camber_pos": round(max_camber_pos), 
        "thickness": round(max_thickness),
        "coords": (x_range, y_upper, y_lower)
    }

# ========================= MODEL BUILD AND TRAIN ===============================
if __name__ == "__main__":
    # 1. Prepare the data
    print("Loading data...")
    train_dataset, x_mean, x_std = create_aerospace_dataset("airfoil_dataset.csv", batch_size=32)
    
    print("Saving normalization arrays...")
    np.save("x_mean.npy", x_mean)
    np.save("x_std.npy", x_std)     

    # 2. Build the model
    model = build_surrogate_model()

    # 3. Train it
    print("Training the Aerodynamic Surrogate...")
    history = model.fit(train_dataset, epochs=200)
    
    # 4. Save it
    model.save("aerodynamic_surrogate.keras")
    print("Model saved successfully!")
# ================================================================================


import os
import subprocess

def get_xfoil_cl_cd(filepath, alpha, reynolds):
    polar_file = "temp_single_polar.txt"
    if os.path.exists(polar_file):
        os.remove(polar_file)

    # XFOIL Command Sequence for a SINGLE alpha
    commands = [
        "PLOP", "G F",         # Disable graphics
        "",                    
        f"LOAD {filepath}",    # Load coordinates
        "CustomAirfoil",       # Handle naming prompt
        "PANE",                # Smooth panels
        "OPER",                
        f"Visc {reynolds}",    # Set Reynolds number
        "ITER 200",            # High iterations for stubborn boundary layers
        "PACC",                # Start recording to file
        polar_file,            
        "",                    
        f"ALFA {alpha}",       # <--- Solve for just this one Angle of Attack
        "",                    
        "QUIT"                 
    ]
    
    command_string = "\n".join(commands) + "\n"

    # Execute XFOIL
    try:
        process = subprocess.Popen(
            ['xvfb-run', 'xfoil'], 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # 15 seconds is plenty for a single alpha calculation
        process.communicate(input=command_string, timeout=15) 
        
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate() 
        print(f"TIMEOUT: XFOIL failed to converge for {filepath} at alpha={alpha}.")
        return None, None

    # Parse the resulting text file (No Pandas needed)
    cl, cd = None, None
    if os.path.exists(polar_file):
        with open(polar_file, 'r') as f:
            lines = f.readlines()
            
            # XFOIL headers take up about 12 lines. We search for the actual data.
            for line in lines:
                parts = line.strip().split()
                # A valid data row has at least 3 numbers and doesn't contain header text
                if len(parts) >= 3 and "alpha" not in line and "---" not in line:
                    try:
                        # Ensure we are looking at the row for our requested alpha
                        row_alpha = float(parts[0])
                        if abs(row_alpha - alpha) < 0.1: 
                            cl = float(parts[1])
                            cd = float(parts[2])
                            break # We found it, stop searching
                    except ValueError:
                        pass # Skip lines that can't be converted to floats
                        
        #os.remove(polar_file) # Clean up
        
    return cl, cd
