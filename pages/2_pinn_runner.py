import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
import pinn_runner.pinn as pn

# --- PAGE CONFIG ---
st.set_page_config(page_title="PINN Runner", layout="wide")

st.title("PINN Runner")
st.markdown("""
**Solve differential equations using Physics-Informed Neural Networks!**
Configure your network architecture, select a physical system, and observe the training convergence.
""")

st.divider()

# ==========================================
# 1. PINN CONFIGURATION
# ==========================================
st.subheader("1. PINN Structure")
# pinn structure
col1, col2, col3 = st.columns(3)
with col1:
    layers = st.number_input("Hidden Layers", min_value=1, max_value=10, value=3, step=1)
with col2:
    neurons = st.number_input("Neurons per Layer", min_value=10, max_value=200, value=50, step=10)
with col3:
    epochs = st.number_input("Epochs", min_value=1000, max_value=50000, value=5000, step=1000)
# learning rate
col4, col5 = st.columns(2)
with col4:
    learnrate = st.slider("Initial Learning Rate", 0.0001, 0.01, 0.001, step=0.0001, format="%.4f")
with col5:
    decayrate = st.slider("Decay Rate (per 1000 steps)", 0.80, 0.99, 0.95)


st.divider()
# ==========================================
# 2. SYSTEM SELECTION & PARAMETERS
# ==========================================
st.subheader("2. Physics Model Selection")
system_name = st.selectbox(
    "Choose a Physical System",
    ["Newton's Law of Cooling", "Simple Harmonic Oscillator"]
)

if system_name == "Newton's Law of Cooling":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(r"""
            **Newton's Law of Cooling** models the heat transfer between an object and its environment.
            $$
            \begin{aligned}
            \frac{dT}{dt} &= -k(T - T_{surr})
            \end{aligned}
            $$
            """)
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("Model Parameters")
                T_init = st.number_input("Initial Temp (T₀)", value=250.0, step=10.0)
                T_surr = st.number_input("Ambient Temp (T_surr)", value=27.0, step=1.0)
                k_cool = st.number_input("Cooling Constant (k)", min_value=0.01, value=0.45, step=0.05)
        
        with col2:
            with st.container(border=True):
                st.subheader("Loss Weights")
                w_pde = st.number_input("PDE Weight", min_value=1.0, value=1.0, step=1.0)
                w_bc = st.number_input("B.C. Weight", min_value=1.0, value=1.0, step=1.0)
                st.markdown("B.C. weight acts like an anchor of the starting point, while PDE weight acts like a pathfinder of the solution.")

elif system_name == "Simple Harmonic Oscillator":
        st.markdown(r"""
        The **Simple Harmonic Oscillator** models undamped periodic motion, foundational for structural vibration analysis.
        $$
        \begin{aligned}
        m\ddot{x} + kx &= 0 \\
        x(0) = x_0, \quad \dot{x}(0) &= v_0
        \end{aligned}
        $$
        """)
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("Model Parameters")
                m_mass = st.number_input("Mass (m)", min_value=0.1, value=1.0, step=0.1)
                k_stiff = st.number_input("Stiffness (k)", min_value=0.1, value=10.0, step=1.0)
                x0 = st.number_input("Initial Position (x₀)", value=1.0, step=0.1)
                v0 = st.number_input("Initial Velocity (v₀)", value=0.0, step=0.1)

        with col2:
            with st.container(border=True):
                st.subheader("Loss Weights")
                w_pde = st.number_input("PDE Weight", min_value=1.0, value=1.0, step=1.0)
                w_bc = st.number_input("B.C. Weight", min_value=1.0, value=1.0, step=1.0)
                st.markdown("B.C. loss acts like an anchor of the starting point, while PDE loss acts like a pathfinder of the solution.")
        

st.divider()
# ==========================================
# 3. RUN SIMULATION & TF GRAPH
# ==========================================

st.subheader("Train with...")
mode = st.segmented_control(
    "Engine",
    options=["TensorFlow", "PyTorch", "DeepXDE"],
    default="TensorFlow"
)

if st.button("Train PINN", type="primary"):
    tf.keras.backend.clear_session()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    t_test = np.linspace(0, 10, 500).reshape(-1, 1)
    t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
    
    loss_history = []
    epochs_sampled = []
    losses_sampled = []
    # ------------------------------------------
    if system_name == "Newton's Law of Cooling":
        params = (T_init, T_surr, k_cool)
    else:
        params = (m_mass, k_stiff, x0, v0)
        
    # ------------------------ TENSORFLOW ------------------------
    model = pn.build_pinn(layers, neurons)
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(learnrate, 1000, decayrate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
    
    train_step = pn.get_train_step(system_name, model, optimizer, params, (w_pde, w_bc))
    # ------------------------ PYTORCH ------------------------
    pass
    # ------------------------ DEEPXDE ------------------------
    
    
    # Dummy
    _ = train_step()
    
    start_time = time.time()
    for epoch in range(epochs + 1):
        loss_val = float(train_step())
        loss_history.append(loss_val)
        
        if epoch % 1000 == 0:
            epochs_sampled.append(epoch)
            losses_sampled.append(loss_val)
            progress_bar.progress(epoch / epochs)
            status_text.write(f"**Training...** Epoch {epoch}/{epochs} | Loss: `{loss_val:.6e}`")
            
    train_time = time.time() - start_time
    status_text.success(f"✅ Training Complete in {train_time:.2f} seconds! Final Loss: `{loss_history[-1]:.6e}`")



if st.button('Grid Search', type='primary'):
    # w_pde = 1
    w_bc_options = [1, 10, 100, 1000]
    lr_options = [0.001, 0.00001]

    pass


    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    st.subheader("3. Training Results")
    
    pred_vals = model(t_test_tensor).numpy()
    
    # Analytical Soln
    if system_name == "Newton's Law of Cooling":
        exact_vals = T_surr + (T_init - T_surr) * np.exp(-k_cool * t_test)
        y_label = "Temperature (T)"
    elif system_name == "Simple Harmonic Oscillator":
        omega = np.sqrt(k_stiff / m_mass)
        exact_vals = x0 * np.cos(omega * t_test) + (v0 / omega) * np.sin(omega * t_test)
        y_label = "Position (x)"

    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(t_test, exact_vals, 'k-', linewidth=3, label="Analytical Exact", alpha=0.4)
        ax1.plot(t_test, pred_vals, 'r--', linewidth=2, label="PINN Predicted")
        ax1.set_title("PINN - Prediction vs Reality")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(y_label)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
    with plot_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.bar(epochs_sampled, losses_sampled, width=600, color='#1f77b4', alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_title("Loss (Sampled every 1000 Epochs)")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Total Loss (Log Scale)")
        ax2.set_xticks(epochs_sampled)
        ax2.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig2)