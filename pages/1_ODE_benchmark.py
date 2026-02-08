import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# --- PATH SETUP ---
# Add the parent directory to path so we can import the 'engine' package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from ode_benchmark import library as lib
    from ode_benchmark import solvers as sol
except ImportError:
    st.error("⚠️ Engine not found! Make sure your 'engine' folder contains __init__.py, library.py, and solvers.py")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="ODE Benchmark", page_icon="📈", layout="wide")

st.title("ODE Solver Benchmark Suite")
st.markdown("""
**Compare numerical integration methods** against high-fidelity reference solutions.
Select your physics model, configure the solver settings, and analyze the stability.
""")

st.divider()

# ==========================================
# 1. SYSTEM SELECTION
# ==========================================
st.subheader("1. Physics Model Selection")
system_name = st.selectbox(
    "Choose an ODE System",
    ["Damped Oscillator", "Van der Pol", "Duffing Oscillator", "Lorenz System"]
)

# ==========================================
# 2. SOLVER SETTINGS
# ==========================================
st.subheader("2. Solver Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    methods = st.multiselect(
        "Numerical Methods (Max 3)",
        ["Forward Euler", "Runge-Kutta 4 (RK4)"],
        default=["Forward Euler", "Runge-Kutta 4 (RK4)"]
    )
    if len(methods) > 3:
        st.warning("Please select a maximum of 3 methods.")
        st.stop()

with col2:
    h = st.number_input("Step Size (h)", min_value=0.0001, max_value=1.0, value=0.01, format="%.4f")

with col3:
    t_max = st.number_input("Simulation Time (sec)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

# ==========================================
# 3. MODEL PARAMETERS
# ==========================================
st.subheader("3. Model Parameters")
params = ()
init_cond = []

if system_name == "Damped Oscillator":
    c1, c2, c3 = st.columns(3)
    m = c1.number_input("Mass (m)", value=1.0)
    c = c2.number_input("Damping (c)", value=0.5)
    k = c3.number_input("Stiffness (k)", value=10.0)
    params = (m, c, k)
    
    ic1, ic2 = st.columns(2)
    x0 = ic1.number_input("Initial Position (x)", value=1.0)
    v0 = ic2.number_input("Initial Velocity (v)", value=0.0)
    init_cond = [x0, v0]
    
    # Initialize System Object
    system = lib.DampedOscillator(init_cond, params)

elif system_name == "Van der Pol":
    mu = st.number_input("Non-linearity (μ)", value=2.0)
    params = (mu,)

    st.caption("Initial Conditions")
    ic1, ic2 = st.columns(2)
    x0 = ic1.number_input("Position (x)", value=2.0)
    v0 = ic2.number_input("Velocity (v)", value=0.0)
    init_cond = [x0, v0]

    system = lib.VanderPol(init_cond, params)

elif system_name == "Duffing Oscillator":
    c1, c2, c3, c4, c5 = st.columns(5)
    delta = c1.number_input("Damping (δ)", value=0.1)
    alpha = c2.number_input("Stiffness (α)", value=1.0)
    beta = c3.number_input("Non-linear (β)", value=5.0)
    gamma = c4.number_input("Force (γ)", value=0.3)
    omega = c5.number_input("Freq (ω)", value=1.0)
    params = (delta, alpha, beta, gamma, omega)

    st.caption("Initial Conditions")
    ic1, ic2 = st.columns(2)
    x0 = ic1.number_input("Position (x)", value=0.0)
    v0 = ic2.number_input("Velocity (v)", value=0.0)
    init_cond = [x0, v0]

    system = lib.DuffingOscillator(init_cond, params)

elif system_name == "Lorenz System":
    c1, c2, c3 = st.columns(3)
    sigma = c1.number_input("Sigma (σ)", value=10.0)
    rho = c2.number_input("Rho (ρ)", value=28.0)
    beta_l = c3.number_input("Beta (β)", value=8.0/3.0)
    params = (sigma, rho, beta_l)

    st.caption("Initial Conditions")
    ic1, ic2, ic3 = st.columns(3)
    x0 = ic1.number_input("x", value=1.0)
    y0 = ic2.number_input("y", value=1.0)
    z0 = ic3.number_input("z", value=1.0)
    init_cond = [x0, y0, z0]

    # Assuming you add Lorenz later, otherwise this might fail if not in library.py
    # Use a try/except or placeholder if not yet implemented
    try:
        system = lib.LorenzSystem(init_cond, params)
    except AttributeError:
        st.error("Lorenz System not found in library.py yet!")
        st.stop()

st.divider()

# ==========================================
# 4. RUN SIMULATION
# ==========================================
if st.button("Run Benchmark", type="primary"):
    
    # 1. Setup Time and Reference
    t_eval = np.arange(0, t_max, h)

    # reference soln (RK45)
    with st.spinner("Calculating High-Fidelity Reference..."):
        ref_sol = sol.RK45(system, (0, t_max), t_eval)  # (steps, states)
    
    # 2. Run Numerical Solvers
    results = {}
    errors = {}
    scores = []

    for method in methods:
        y_curr = np.array(init_cond)
        y_history = [y_curr]
        
        # Select function from solvers.py
        solve_func = None
        if method == "Forward Euler":
            solve_func = sol.forward_euler
        elif method == "Runge-Kutta 4 (RK4)":
            solve_func = sol.RK4
        
        # Integration Loop
        for i, t in enumerate(t_eval[:-1]):
            y_curr = solve_func(system, y_curr, t, h)
            y_history.append(y_curr)
            
        results[method] = np.array(y_history)
        
        # Calculate Error (Euclidean distance at each step)
        # We compare against ref_sol. ref_sol might need transposing depending on your library
        # Assuming ref_sol is (steps, states)
        
        # Safety check for shape mismatch
        n_steps = len(results[method])
        ref_trimmed = ref_sol[:n_steps]
        
        abs_error = np.linalg.norm(results[method] - ref_trimmed, axis=1)
        errors[method] = abs_error
        
        # Calc Scores
        mae = np.mean(abs_error)
        log_mae = -np.log10(mae) if mae > 0 else 16.0
        scores.append({
            "Method": method,
            "MAE": f"{mae:.2e}",
            "Log10 Score": f"{log_mae:.2f}",
            "Steps": n_steps
        })

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================
    
    st.subheader("4. Benchmark Results")
    
    # Create Layout
    plot_col, score_col = st.columns([3, 1])
    
    with plot_col:
        # --- PLOT 1: Solution curve ---
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t_eval, ref_sol[:, 0], 'k-', linewidth=3, label="Reference (RK45)", alpha=0.5)
        
        colors = {"Forward Euler": "blue", "Runge-Kutta 4 (RK4)": "red"}
        for method in methods:
            ax1.plot(t_eval, results[method][:, 0], label=method, color=colors.get(method, "green"), linestyle='dashed') # green is default
            
        ax1.set_title(f"{system_name}: Position vs Time")
        ax1.set_ylabel("Position")
        ax1.set_xlabel("Time (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # --- PLOT 2: Log Error ---
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        for method in methods:
            ax2.semilogy(t_eval, errors[method] + 1e-16, label=f"{method} Error", color=colors.get(method, "green"))
            
        ax2.set_title("Global Truncation Error (Log Scale)")
        ax2.set_ylabel("Error |y - y_ref|")
        ax2.set_xlabel("Time (s)")
        ax2.legend()
        ax2.grid(True, which="both", alpha=0.3)
        st.pyplot(fig2)

    with score_col:
        st.markdown("### Reports")
        df_scores = pd.DataFrame(scores)
        df_scores = df_scores.sort_values(by="Log10 Score", ascending=False)
        st.dataframe(df_scores, hide_index=True)
        
        st.info("""
        **Score Guide:**
        * **< 2.0**: Low Accuracy
        * **2.0 - 7.0**: Engineering Grade
        * **> 8.0**: High Precision
        """)