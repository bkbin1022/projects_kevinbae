import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="PINN Runner", layout="wide")

st.title("PINN Solver Benchmark Suite")
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
def build_model(num_layers, num_neurons):
    model = tf.keras.Sequential()
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units=num_neurons, activation="tanh"))
    model.add(tf.keras.layers.Dense(units=1))
    return model

if st.button("Train PINN", type="primary"):
    tf.keras.backend.clear_session()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    t_train = tf.convert_to_tensor(np.linspace(0, 10, 100).reshape(-1, 1), dtype=tf.float32)
    t_test = np.linspace(0, 10, 500).reshape(-1, 1)
    t_test_tensor = tf.convert_to_tensor(t_test, dtype=tf.float32)
    
    model = build_model(layers, neurons)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = learnrate,
        decay_steps = 1000,
        decay_rate = decayrate
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    loss_history = []
    epochs_sampled = []
    losses_sampled = []
    
    # Train loop
    if system_name == "Newton's Law of Cooling":
        t_bc = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
        T_bc = tf.convert_to_tensor([[T_init]], dtype=tf.float32)
        
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                # PDE Loss
                with tf.GradientTape() as tape_pde:
                    tape_pde.watch(t_train)
                    T_pred = model(t_train)
                dT_dt = tape_pde.gradient(T_pred, t_train)
                pde_residual = dT_dt + k_cool * (T_pred - T_surr)
                loss_pde = tf.reduce_mean(tf.square(pde_residual))
                
                # BC Loss
                T_bc_pred = model(t_bc)
                loss_bc = tf.reduce_mean(tf.square(T_bc_pred - T_bc))
                
                # Total Loss
                total_loss = w_pde * loss_pde + w_bc * loss_bc
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss

    elif system_name == "Simple Harmonic Oscillator":
        t_bc = tf.convert_to_tensor([[0.0]], dtype=tf.float32)
        x_bc = tf.convert_to_tensor([[x0]], dtype=tf.float32)
        v_bc = tf.convert_to_tensor([[v0]], dtype=tf.float32)
        omega_sq = k_stiff / m_mass
        
        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                # 1. PDE Loss 
                with tf.GradientTape() as tape1:
                    tape1.watch(t_train)
                    with tf.GradientTape() as tape2:
                        tape2.watch(t_train)
                        x_pred = model(t_train)
                    dx_dt = tape2.gradient(x_pred, t_train)
                d2x_dt2 = tape1.gradient(dx_dt, t_train)
                
                pde_residual = d2x_dt2 + omega_sq * x_pred
                loss_pde = tf.reduce_mean(tf.square(pde_residual))
                
                # 2. BC Loss (Position and Velocity)
                with tf.GradientTape() as tape_bc:
                    tape_bc.watch(t_bc)
                    x_bc_pred = model(t_bc)
                dx_dt_bc = tape_bc.gradient(x_bc_pred, t_bc)
                
                loss_bc_pos = tf.reduce_mean(tf.square(x_bc_pred - x_bc))
                loss_bc_vel = tf.reduce_mean(tf.square(dx_dt_bc - v_bc))
                loss_bc = loss_bc_pos + loss_bc_vel
                
                total_loss = w_pde * loss_pde + w_bc * loss_bc
                
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return total_loss

    # --- EXECUTE TRAINING ---
    # Dummy pass to build variables
    _ = train_step()
    
    start_time = time.time()
    for epoch in range(epochs + 1):
        loss_val = float(train_step())
        loss_history.append(loss_val)
        
        if epoch % 1000 == 0:
            epochs_sampled.append(epoch)
            losses_sampled.append(loss_val)
            progress_bar.progress(epoch / epochs)
            status_text.write(f"**Training...** Epoch {epoch}/{epochs} | Loss: `{loss_val}`")
            
    train_time = time.time() - start_time
    status_text.success(f"✅ Training Complete in {train_time:.2f} seconds! Final Loss: `{loss_history[-1]}`")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    st.subheader("3. Training Results")
    
    # Generate Predictions
    pred_vals = model(t_test_tensor).numpy()
    
    # Generate Analytical Solutions
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
        ax1.set_title(f"PINN - Prediction vs Reality")
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
        ax2.set_xticks(epochs_sampled) # Ensure x-ticks align with the bars
        ax2.grid(True, axis='y', alpha=0.3)
        st.pyplot(fig2)