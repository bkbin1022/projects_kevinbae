import streamlit as st

st.set_page_config(page_title="KAC Lab", page_icon="🚀")

st.title("Kwangbin's Aerospace Computing Lab")
st.subheader("Welcome")

st.markdown("""
The KAC Lab is a personal lab experimenting with all kinds of aerospace fields with Python.<br>
The lab is especially focused on numerical methods, neural networks, and design optimization.<br>
Projects are started when I get interested into a specific field. Because of this behavior, some completely random topic might come out!

Currently, the lab has completed the following projects.            
* **ODE Benchmark:** Benchmark to compare ODE solvers (Euler, RK4, Heun's, ...).    <a href="requirements.txt">doc</a>
* **PINN:** Build and train Physics-Informed Neural Networks using Tensorflow, Pytorch or DeepXDE.    <a href="requirements.txt">doc</a>
* **TopOpt:** Optimize airfoil structure using topology optimization. <a href="requirements.txt">doc</a>
* **Netfoil:** Predict lift and drag coefficients instantaneously using trained neural networks.    <a href="requirements.txt">doc</a>

👈 Select a project from the sidebar to begin.
""", unsafe_allow_html=True)