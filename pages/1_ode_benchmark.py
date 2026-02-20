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
    ["Damped Oscillator", "Van der Pol", "Duffing Oscillator"]
)

col1, col2, col3 = st.columns([1, 2, 1])
if system_name == "Damped Oscillator":

    with col2:
        st.markdown(r"""
                    In a **damped system**, energy dissipation (often due to friction or air resistance) 
                    causes the amplitude of oscillations to decay over time until the system reaches equilibrium.
                    """)
        st.markdown(r"""
                    $$
                    \begin{aligned}
                    \text{Governing Equation:} \quad & m\ddot{x} + c\dot{x} + kx = 0 \\
                    \text{State-Space Form:} \quad & \dot{x} = v \\
                    & \dot{v} = \frac{-cv - kx}{m}
                    \end{aligned}
                    $$
                    """)
elif system_name == "Van der Pol":
    with col2:
        st.markdown(r"""
                    The **Van der Pol oscillator** is a non-linear oscillator with non-conservative damping. 
                    It settles into a stable **limit cycle**, where the system repeats the same vibration 
                    pattern regardless of its initial starting point.
                    """)

        st.markdown(r"""
                    $$
                    \begin{aligned}
                    \text{Governing Equation:} \quad & \ddot{x} - \mu(1 - x^2)\dot{x} + x = 0 \\
                    \text{State-Space Form:} \quad & \dot{x} = v \\
                    & \dot{v} = \mu(1 - x^2)v - x
                    \end{aligned}
                    $$
                    """)
elif system_name == "Duffing Oscillator":
    with col2:
        st.markdown(r"""
                    The **Duffing oscillator** models a spring-mass system with a non-linear restoring force. 
                    It is frequently used in aerospace research to simulate structural vibrations 
                    and "snap-through" buckling in thin-walled structures.
                    """)

        st.markdown(r"""
                    $$
                    \begin{aligned}
                    \text{Governing Equation:} \quad & \ddot{x} + \delta\dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t) \\
                    \text{State-Space Form:} \quad & \dot{x} = v \\
                    & \dot{v} = \gamma \cos(\omega t) - \delta v - \alpha x - \beta x^3
                    \end{aligned}
                    $$
                    """)



# ==========================================
# 2. SOLVER SETTINGS
# ==========================================
st.subheader("2. Solver Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    methods = st.multiselect(
        "Numerical Methods (Max 3)",
        ["Forward Euler", "Runge-Kutta 4 (RK4)", "Heun's Method"],
        default=["Forward Euler", "Runge-Kutta 4 (RK4)"]
    )

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
    m = c1.number_input("Mass (m)", min_value=0.0001, value=1.0, step=1.0)
    c = c2.number_input("Damping (c)", value=0.5, step=0.1)
    k = c3.number_input("Stiffness (k)", value=10.0, step=1.0)
    params = (m, c, k)
    
    ic1, ic2 = st.columns(2)
    x0 = ic1.number_input("Initial Position (x)", value=1.0, step=1.0)
    v0 = ic2.number_input("Initial Velocity (v)", value=0.0, step=1.0)
    init_cond = [x0, v0]
    
    # Initialize System Object
    system = lib.DampedOscillator(init_cond, params)

elif system_name == "Van der Pol":
    mu = st.number_input("Non-linearity (μ)", value=2.0, step=1.0)
    params = (mu,)

    st.caption("Initial Conditions")
    ic1, ic2 = st.columns(2)
    x0 = ic1.number_input("Position (x)", value=2.0, step=1.0)
    v0 = ic2.number_input("Velocity (v)", value=0.0, step=1.0)
    init_cond = [x0, v0]

    system = lib.VanderPol(init_cond, params)

elif system_name == "Duffing Oscillator":
    c1, c2, c3, c4, c5 = st.columns(5)
    delta = c1.number_input("Damping (δ)", value=0.1, step=0.1)
    alpha = c2.number_input("Stiffness (α)", value=1.0, step=1.0)
    beta = c3.number_input("Non-linear (β)", value=5.0, step=1.0)
    gamma = c4.number_input("Force (γ)", value=0.3, step=0.1)
    omega = c5.number_input("Freq (ω)", value=1.0, step=1.0)
    params = (delta, alpha, beta, gamma, omega)

    st.caption("Initial Conditions")
    ic1, ic2 = st.columns(2)
    x0 = ic1.number_input("Position (x)", value=0.0, step=1.0)
    v0 = ic2.number_input("Velocity (v)", value=0.0, step=1.0)
    init_cond = [x0, v0]

    system = lib.DuffingOscillator(init_cond, params)

st.divider()

# ==========================================
# 4. RUN SIMULATION
# ==========================================
if st.button("Run Benchmark", type="primary"):
    
    # 1. Setup Time and Reference
    t_eval = np.arange(0, t_max, h)

    # reference soln (RK45)
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
        elif method == "Heun's Method":
            solve_func = sol.heun_method
        
        # Integration Loop
        for i, t in enumerate(t_eval[:-1]):
            y_curr = solve_func(system, y_curr, t, h)
            y_history.append(y_curr)
            
        results[method] = np.array(y_history)

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
    plot_col, score_col = st.columns([2, 1])
    
    with plot_col:
        # --- PLOT 1: Solution curve ---
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(t_eval, ref_sol[:, 0], 'k-', linewidth=3, label="Reference (RK45)", alpha=0.5)
        
        colors = {"Forward Euler": "blue", "Runge-Kutta 4 (RK4)": "red", "Heun's Method": "green"}
        for method in methods:
            ax1.plot(t_eval, results[method][:, 0], label=method, color=colors.get(method), linestyle='dashed') 
            
        ax1.set_title(f"{system_name}: Position vs Time")
        ax1.set_ylabel("Position")
        ax1.set_xlabel("Time (s)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # --- PLOT 2: Log Error ---
        fig2, ax2 = plt.subplots(figsize=(8, 3))
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