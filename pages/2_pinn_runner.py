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
col1, col2, col3 = st.columns(3)
with col1:
    layers = st.number_input("Hidden Layers", min_value=1, max_value=10, value=3, step=1)
with col2:
    neurons = st.number_input("Neurons per Layer", min_value=10, max_value=200, value=50, step=10)
with col3:
    epochs = st.number_input("Epochs", min_value=1000, max_value=50000, value=5000, step=1000)

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

col2_1, col2_2, col2_3 = st.columns([1, 2, 1])
if system_name == "Newton's Law of Cooling":

    with col2_2:
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
            params = (T_init, T_surr, k_cool)
    
    with col2:
        with st.container(border=True):
            st.subheader("Loss Weights")
            w_pde = st.number_input("PDE Weight", min_value=1.0, value=1.0, step=1.0)
            w_bc = st.number_input("B.C. Weight", min_value=1.0, value=1.0, step=1.0)

elif system_name == "Simple Harmonic Oscillator":
    with col2_2:
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
            params = (m_mass, k_stiff, x0, v0)

    with col2:
        with st.container(border=True):
            st.subheader("Loss Weights")
            w_pde = st.number_input("PDE Weight", min_value=1.0, value=1.0, step=1.0)
            w_bc = st.number_input("B.C. Weight", min_value=1.0, value=1.0, step=1.0)

st.divider()

# ==========================================
# 3. ADDITIONAL FACTORS
# ==========================================
st.subheader("3. Additional Factors")

st.markdown("Tune hyperparameters (Loss weight, learning rate, neuron count) for better PINNs!")
if st.button("Tune Hyperparameters", type="primary"):
    # 1. WEIGHTS
    results_w = {}
    with st.spinner("Optimizing Loss Weight..."):
        for pde_weight in [0.01, 0.1, 1.0, 10.0]:
            tf.keras.backend.clear_session()
            model = pn.build_pinn(layers, neurons) 
            lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(learnrate, 1000, decayrate)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)
            train_step = pn.get_train_step(system_name, model, optimizer, params, (pde_weight, 1.0)) # bc_weight fixed to 1.0

            for epoch in range(500):
                loss = float(train_step()) 
            results_w[pde_weight] = loss
            best_weight = min(results_w, key=results_w.get)
        st.markdown(f"{results_w}")
        st.markdown(f"Best weight: {best_weight}")
    # 2. INITIAL LEARNRATE
    results_lr = {}
    with st.spinner("Optimizing initial learning rate..."):
        for lr in [0.0001, 0.001, 0.01, 0.1]:
            tf.keras.backend.clear_session()
            model = pn.build_pinn(layers, neurons) 
            lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(lr, 1000, decayrate)  # Optimizing
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)                    # these
            train_step = pn.get_train_step(system_name, model, optimizer, params, (best_weight, 1.0)) # pde_weight from #1
            for epoch in range(500):
                loss = float(train_step()) 
            results_lr[lr] = loss
            best_lr = min(results_lr, key=results_lr.get)
        st.markdown(f"{results_lr}")
        st.markdown(f"Best lr: {best_lr}")
    # 3. NEURONS
    results_nrns = {}
    with st.spinner("Optimizing neuron count..."):
        for nrns in [20, 40, 60, 80]:
            tf.keras.backend.clear_session()
            lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(best_lr, 1000, decayrate)   # best_lr from #2
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)       
            model = pn.build_pinn(layers, nrns)              # Optimizing this     
            train_step = pn.get_train_step(system_name, model, optimizer, params, (best_weight, 1.0)) # pde_weight from #1
            for epoch in range(500):
                loss = float(train_step()) 
            results_nrns[nrns] = loss
            best_nrns = min(results_nrns, key=results_nrns.get)
        st.markdown(f"{results_nrns}")
        st.markdown(f"Best neurons: {best_nrns}")
    df_results = pd.DataFrame({
        "PDE Weight": results_w.keys(),
        "Final Loss": [f"{v:,.2f}" for v in results_w.values()]
        })

st.dataframe(df_results, hide_index=True)

st.markdown("Change Optimizer. Default: Adam")

st.divider()



# ==========================================
# 4. TRAINING ENGINE
# ==========================================
st.subheader("4. Execution")
mode = st.segmented_control(
    "Engine",
    options=["TensorFlow"],
    default="TensorFlow"
)

if st.button("Train PINN", type="primary"):
    loss_history = []
    epochs_sampled = []
    losses_sampled = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Reset NN
    tf.keras.backend.clear_session()
    t_test = np.linspace(0, 10, 500).reshape(-1, 1)
    t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
    model = pn.build_pinn(layers, neurons)
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(learnrate, 1000, decayrate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched)

    train_step = pn.get_train_step(system_name, model, optimizer, params, (w_pde, w_bc))

    # Optimization Loop
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
    status_text.success(f"✅ Training Complete in {train_time:.2f} seconds! Final Loss: `{loss_history[-1]}`")

    # Generate Predictions
    pred_vals = model(t_test_tensor).numpy()

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    st.subheader("5. Results & Analysis")
    
    # Analytical Solution Calculation
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
        ax1.set_title("Prediction vs Ground Truth")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel(y_label)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
    with plot_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(epochs_sampled, losses_sampled, width=600, color='#1f77b4', alpha=0.8)
        ax2.set_yscale('log')
        ax2.set_title("Loss Convergence")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Total Loss (Log Scale)")
        ax2.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig2)