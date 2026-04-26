import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import projects.netfoil.library as nf

@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model("datafiles/aerodynamic_surrogate.keras")
        x_mean = np.load("datafiles/x_mean.npy")
        x_std = np.load("datafiles/x_std.npy")
        return model, x_mean, x_std
    except Exception as e:
        st.error(f"Asset Error: {e}")
        return None, None, None

model, x_mean, x_std = load_assets()
df = pd.read_csv("datafiles/airfoil_dataset.csv", dtype={'NACA': str})

# --- UI SETUP ---
st.set_page_config(page_title="Netfoil")
st.title("Neural Airfoil Predictor")

# --- INITIALIZE VARIABLES (Prevents NameErrors) ---
naca_input = "2412" 
uploaded_file = None
camber, camber_pos, thickness = 0.02, 0.4, 0.12

# --- INPUT SECTION ---
st.subheader("1. Geometry Configuration")
input_mode = st.radio("Input Method:", ["Manual NACA Code", "Upload Coordinates (.dat/.txt)"])

if input_mode == "Manual NACA Code":
    naca_input = st.text_input("NACA 4-Digit", value="2412", max_chars=4)
    if len(naca_input) == 4 and naca_input.isdigit():
        camber = float(naca_input[0]) / 100.0
        camber_pos = float(naca_input[1]) / 10.0
        thickness = float(naca_input[2:4]) / 100.0
    with st.expander("View Supported NACA Range"):
        st.write("""
        **The Neural Network was trained on the following NACA 4-Digit configurations:**
        * **Symmetrical:** 0006 to 0024
        * **Low Camber:** 2408 to 2424
        * **High Camber:** 4410 to 4421 and 6410 to 6421
        * **Reynolds Numbers:** 100k, 500k, 1M, 5M, 10M
        """)
    
        # Optionally show a sorted list of unique NACAs from your actual CSV
        unique_nacas = sorted(df['NACA'].unique())
        st.caption(f"Total unique airfoils in database: {len(unique_nacas)}")
        st.write(", ".join(unique_nacas))
else:
    uploaded_file = st.file_uploader("Upload Airfoil", type=['dat', 'txt'])
    if uploaded_file:
        geom = nf.parse_airfoil_file(uploaded_file)
        if geom:
            camber, camber_pos, thickness = geom['camber']/100, geom['camber_pos']/10, geom['thickness']/100
            naca_input = f"{geom['camber']}{geom['camber_pos']}{geom['thickness']:02d}"
            st.line_chart(np.array([geom['coords'][1], geom['coords'][2]]).T)

st.subheader("2. Flight Conditions")
col_a, col_b = st.columns(2)
with col_a:
    alpha = st.slider("Alpha (deg)", -5.0, 15.0, 5.0, 1.0)
with col_b:
    reynolds = st.select_slider("Reynolds Number", 
                                options=[1e5, 5e5, 1e6, 5e6, 1e7], 
                                value=1e6, format_func=lambda x: f"{x:,.0f}")

# --- PREDICTION LOGIC ---
if st.button('Predict', type='primary'):
        # --- 1. ALWAYS DO THE NEURAL NETWORK PREDICTION ---
        raw_input = np.array([[camber, camber_pos, thickness, reynolds, alpha]])
        scaled_input = (raw_input - x_mean) / x_std
        predictions = model.predict(scaled_input, verbose=0)
        cl_final = predictions[0][0]
        cd_log = predictions[0][1]
        cd_final = 10**cd_log
        
        # Display NN Results
        st.subheader(f"Results for NACA {naca_input}")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric(label="Lift Coefficient (CL)", value=f"{cl_final:.4f}")
        with result_col2:
            st.metric(label="Drag Coefficient (CD)", value=f"{cd_final:.5f}")
        
        st.divider()

        # --- 2. CONDITIONAL XFOIL COMPARISON ---
        if input_mode == "Upload Coordinates (.dat/.txt)":
            st.warning("⚠️ XFOIL comparison not supported for uploaded airfoils!")
        else:
            # Only run this if we are in Manual NACA mode

            xfoil_data = df.loc[
                (df['NACA'] == naca_input) & (df['Re'] == reynolds) & (df['alpha'] == alpha)
            ]

            if not xfoil_data.empty:
                cl_xfoil = xfoil_data['CL'].item()
                cd_xfoil = xfoil_data['CD'].item()

                st.subheader("Analysis")
                analysis_col1, analysis_col2 = st.columns(2)

                lift_error = np.abs(cl_final - cl_xfoil)/cl_xfoil * 100
                drag_error = np.abs(cd_final - cd_xfoil)/cd_xfoil * 100
                
                with analysis_col1:
                    st.metric(label="Error (CL)", value=f"{lift_error:.2f}%") 
                    st.markdown(f"XFOIL value: {cl_xfoil:.4f}")
                with analysis_col2:
                    st.metric(label="Error (CD)", value=f"{drag_error:.2f}%")
                    st.markdown(f"XFOIL value: {cd_xfoil:.5f}")
            else:
                st.info("No exact match found in dataset for these flight conditions.")

# --- PLOTTING SECTION ---

        alpha_sweep = np.arange(-5, 16, 1.0)
        batch_in = np.zeros((len(alpha_sweep), 5))
        batch_in[:, 0], batch_in[:, 1], batch_in[:, 2] = camber, camber_pos, thickness
        batch_in[:, 3], batch_in[:, 4] = reynolds, alpha_sweep

        # 1. Get predictions from model
        nn_sweep = model.predict((batch_in - x_mean) / x_std, verbose=0)
        
        # 2. SEPARATE AND TRANSFORM: Convert log10(CD) back to linear CD
        nn_cl = nn_sweep[:, 0]
        nn_cd_linear = 10**(nn_sweep[:, 1]) 

        if input_mode == "Manual NACA Code":
            # Get ground truth from CSV
            csv_curve = df[(df['NACA'] == naca_input) & (df['Re'] == reynolds)].sort_values('alpha')

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            
            # --- Left Plot: Cl v Alpha ---
            ax[0].plot(alpha_sweep, nn_cl, 'r--', label='Netfoil')
            ax[0].scatter(csv_curve['alpha'], csv_curve['CL'], color='black', label='XFOIL')
            ax[0].set_title("Lift Curve (Cl v α)")
            ax[0].set_xlabel("Angle of Attack (α)") # Fixed label
            ax[0].set_ylabel("Lift Coefficient (Cl)")
            ax[0].legend()
            ax[0].grid(True)

            # --- Right Plot: Cl v Cd (Drag Polar) ---
            # Use the TRANSFORMED linear drag here
            ax[1].plot(nn_cd_linear, nn_cl, 'r--', label='Netfoil') 
            ax[1].scatter(csv_curve['CD'], csv_curve['CL'], color='black', label='XFOIL')
            ax[1].set_title("Drag Polar (Cl v Cd)")
            ax[1].set_xlabel("Drag Coefficient (Cd)") # Fixed label
            ax[1].set_ylabel("Lift Coefficient (Cl)")
            ax[1].grid(True)
            
            plt.tight_layout() # Prevents labels from overlapping
            st.pyplot(fig)