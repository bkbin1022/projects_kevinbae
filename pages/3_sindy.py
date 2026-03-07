import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysindy as ps
from scipy.integrate import solve_ivp
from pysindy.differentiation import SmoothedFiniteDifference

# --- PAGE CONFIG ---
st.set_page_config(page_title="SINDy", layout="wide")

st.title("SINDy")
st.markdown("""
**Find differential equations using SINDy!**
Configure your network architecture, select a physical system, and observe the training convergence.
""")

st.divider()

# ==========================================
# 1. IMPORT EQUATION
# ==========================================
st.subheader("1. Import Equation Data")
uploaded_file = st.file_uploader(
        "Drop your CSV here or click to browse", 
        type=["csv"],
    )

if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df)} rows of data.")
        
        if st.button("Analyze", type="primary"):
            x_train = df['x'].values.reshape(-1, 1) 
            model = ps.SINDy()
            model.fit(x_train, t=0.001)
            st.session_state.discovered_eqs = model.equations()
            st.session_state.data_processed = True

if st.session_state.get("data_processed", False):
    st.success("✅ Analysis Complete!")
    
    st.subheader("Discovered Governing Equations")
    for eq in st.session_state.discovered_eqs:
        st.markdown(eq)



st.divider()

# ==========================================
# 2. Physics Model Selection
# ==========================================
st.subheader("2. Physics Model Selection")
system_name = st.selectbox(
    "Choose a Physical System",
    ["Van der Pol Oscillator", "Simple Harmonic Oscillator"]
)

from pysindy.differentiation import SINDyDerivative

if system_name == "Van der Pol Oscillator":
    library = ps.PolynomialLibrary(degree=3)
    optimizer = ps.STLSQ(threshold=0.1)
    differentiator = ps.FiniteDifference(order=2)

    mu = st.number_input("Non-linearity (μ)", value=2.0, step=1.0)
    dt = st.number_input("Time step", value=0.001, step=0.0001, format="%.4f")
    def vanderpol(t, y, mu=mu):
        x, v = y
        dxdt = v
        dvdt = mu * (1 - x**2) * v - x
        return [dxdt, dvdt]

    t_train = np.arange(0, 1, dt)
    x0_train = [2.0, 0.0]
    t_train_span = (t_train[0], t_train[-1])
    x_train = solve_ivp(
        vanderpol, t_train_span, x0_train, method='LSODA', t_eval=t_train
    ).y.T

if st.button("SINDy", type="primary"):
    model = ps.SINDy(
        feature_library=library,
        optimizer=optimizer,
        differentiation_method=differentiator
        )
    model.fit(x_train, t=dt)
    model.print()
    st.markdown(model.equations())

st.divider()

