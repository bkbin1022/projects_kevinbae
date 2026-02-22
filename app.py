import streamlit as st

st.set_page_config(page_title="Numerical Playground", page_icon="🚀")

st.title("Numerical Playground")
st.subheader("A playground of Python-driven implentations of all kinds of numerical methods")

st.markdown("""
This site showcases my work in:
* **ODE Benchmark:** Benchmark to compare ODE solvers (Euler, RK4, Heun's, ...).
* **PINN:** Build and train Physics-Informed Neural Networks using Tensorflow, Pytorch or DeepXDE.

👈 Select a project from the sidebar to begin.
""")