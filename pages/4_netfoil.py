import streamlit as st
import numpy as np
import tensorflow as tf
# Note: You don't actually need to import netfoil.library here anymore 
# because you are loading the saved .keras model directly!

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neural Airfoil Predictor", page_icon="✈️", layout="centered")

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
        
    alpha = st.slider("Angle of Attack (deg)", min_value=-5.0, max_value=15.0, value=5.0, step=0.5)

    reynolds = st.select_slider(
        "Reynolds Number", 
        options=[100000, 500000, 1000000, 5000000, 10000000],
        value=1000000,
        format_func=lambda x: f"{x:,}"
    )

    if st.button('Predict', type='primary'):
        # --- FEATURE ENGINEERING ---
        camber = float(naca_input[0]) / 100.0
        camber_pos = float(naca_input[1]) / 10.0
        thickness = float(naca_input[2:4]) / 100.0
        raw_input = np.array([[camber, camber_pos, thickness, reynolds, alpha]])

        # --- NORMALIZATION & PREDICTION ---
        scaled_input = (raw_input - x_mean) / x_std
        
        predictions = model.predict(scaled_input, verbose=0)

        cl_final = predictions[0][0]
        cd_final = predictions[0][1]
        
        st.subheader(f"Results for NACA {naca_input}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Lift Coefficient (CL)", value=f"{cl_final:.4f}")
            
        with col2:
            st.metric(label="Drag Coefficient (CD)", value=f"{cd_final:.5f}")
            
        st.divider()

        with st.expander("View Normalized Neural Network Inputs"):
            st.write("These are the scaled values being fed directly into the model:")
            st.json({
                "Scaled Camber": float(scaled_input[0][0]),
                "Scaled Camber Pos": float(scaled_input[0][1]),
                "Scaled Thickness": float(scaled_input[0][2]),
                "Scaled Reynolds": float(scaled_input[0][3]),
                "Scaled Alpha": float(scaled_input[0][4])
            })


# TASK: ADD INDIVIDUAL COMPARISON TO XFOIL (BASE)
#       ADD PLOT OF COMPARISONS
#       MORE INPUTS AND OUTPUTS
#       SEPARATE XFOIL RUNNING CODE FROM library.py TO ANOTHER FILE
#       DRAG IS NOT BEING CAPTURED WELL, ENHANCE MODEL