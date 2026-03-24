import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("airfoil_dataset.csv", dtype={'NACA': str})

# --- PAGE CONFIG ---
st.set_page_config(page_title="Netfoil", page_icon="✈️", layout="centered")

# --- LOAD MODEL & DATA (Cached for speed) ---
@st.cache_resource
def load_assets():
    try:
        model = tf.keras.models.load_model("aerodynamic_surrogate.keras")
        x_mean = np.load("x_mean.npy")
        x_std = np.load("x_std.npy")
        return model, x_mean, x_std
    except Exception as e:
        st.error(f"Error loading files. Did you run your training script first? Details: {e}")
        return None, None, None

model, x_mean, x_std = load_assets()

st.title("Neural Airfoil Predictor")
st.markdown("""
Predict Lift (CL) and Drag (CD) coefficients instantly using a Deep Neural Network trained on XFoil data.
""")
st.divider()

if model is not None:
    st.subheader("Flight Parameters")
        
    naca_input = st.text_input("NACA 4-Digit Code", value="2412", max_chars=4)
    if not naca_input.isdigit() or len(naca_input) != 4:
        st.error("Please enter a valid 4-digit number (e.g., 2412).")
        st.stop()
        
    alpha = st.slider("Angle of Attack (deg)", min_value=-5.0, max_value=15.0, value=5.0, step=1.0)

    reynolds = st.select_slider(
        "Reynolds Number", 
        options=[100000.0, 500000.0, 1000000.0, 5000000.0, 10000000.0],
        value=1000000.0,
        format_func=lambda x: f"{x:,}"
    )
    # NACA GEOMETRY
    camber = float(naca_input[0]) / 100.0
    camber_pos = float(naca_input[1]) / 10.0
    thickness = float(naca_input[2:4]) / 100.0
    raw_input = np.array([[camber, camber_pos, thickness, reynolds, alpha]])

    if st.button('Predict', type='primary'):
        # --- NORMALIZATION & PREDICTION ---
        scaled_input = (raw_input - x_mean) / x_std
        
        predictions = model.predict(scaled_input, verbose=0)

        cl_final = predictions[0][0]
        cd_final = predictions[0][1]

        cl_xfoil = df.loc[
            (df['NACA'] == f"{naca_input}") & 
            (df['Re'] == reynolds) & 
            (df['alpha'] == alpha), 
            'CL'
        ].item()
#WARNING: THERE ARE SOME VALUES THAT XFOIL COULDNT RUN FOR SPECIFIC ALFAS! SOLVE THIS 
        cd_xfoil = df.loc[
            (df['NACA'] == f"{naca_input}") & 
            (df['Re'] == reynolds) & 
            (df['alpha'] == alpha), 
            'CD'
        ].item()
        
        st.subheader(f"Results for NACA {naca_input}")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric(label="Lift Coefficient (CL)", value=f"{cl_final:.4f}")
        with result_col2:
            st.metric(label="Drag Coefficient (CD)", value=f"{cd_final:.5f}")
        st.divider()


        st.subheader(f"Analysis")
        analysis_col1, analysis_col2 = st.columns(2)

        lift_error = np.abs(cl_final - cl_xfoil)/cl_xfoil * 100
        drag_error = np.abs(cd_final - cd_xfoil)/cd_xfoil * 100
        with analysis_col1:
            st.metric(label="Error (CL)", value=f"{lift_error:.2f}%") 
            st.markdown(f"XFOIL value: {cl_xfoil:.4f}")
        with analysis_col2:
            st.metric(label="Error (CD)", value=f"{drag_error:.2f}%")
            st.markdown(f"XFOIL value: {cd_xfoil:.5f}")

        st.divider()

    if st.button("Generate Performance Plots", type="primary"):
        alpha_sweep = np.arange(-5, 16, 1.0)
        
        # Repeat the geometry/Re for all 21 rows
        batch_input = np.zeros((len(alpha_sweep), 5))
        batch_input[:, 0] = camber
        batch_input[:, 1] = camber_pos
        batch_input[:, 2] = thickness
        batch_input[:, 3] = reynolds
        batch_input[:, 4] = alpha_sweep
        
        scaled_batch = (batch_input - x_mean) / x_std
        
        nn_results = model.predict(scaled_batch, verbose=0)
        nn_cl = nn_results[:, 0]
        nn_cd = nn_results[:, 1]
        
        xfoil_data = df.loc[
            (df['NACA'] == naca_input) & (df['Re'] == reynolds)
        ].sort_values('alpha')
                
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        # Lift Curve (CL vs Alpha)
        ax[0].plot(alpha_sweep, nn_cl, 'r--', label='Netfoil')
        ax[0].plot(xfoil_data['alpha'], xfoil_data['CL'], 'ko', label='XFOIL')
        ax[0].set_xlabel("Alpha (deg)")
        ax[0].set_ylabel("CL")
        ax[0].set_title("Lift Curve")
        ax[0].legend()
        ax[0].grid(True)
        
        # Drag Polar (CL vs CD)
        ax[1].plot(nn_cd, nn_cl, 'r--', label='Netfoil')
        ax[1].plot(xfoil_data['CD'], xfoil_data['CL'], 'ko', label='XFOIL')
        ax[1].set_xlabel("CD")
        ax[1].set_ylabel("CL")
        ax[1].set_title("Drag Polar")
        ax[1].grid(True)
        
        st.pyplot(fig)



# TASK: FEATURE OF IMPORTING AIRFOILS
#       ADD PLOT OF COMPARISONS
#       MORE INPUTS AND OUTPUTS
#       SEPARATE XFOIL RUNNING CODE FROM library.py TO ANOTHER FILE
#       DRAG IS NOT BEING CAPTURED WELL, ENHANCE MODEL