import numpy as np
import matplotlib.pyplot as plt



class BaseODE:
    def __init__(self, init_cond, params, name="Generic ODE System"):
        self.y0 = np.array(init_cond)
        self.init_cond = init_cond
        self.params = params
        self.name = name

    def derivs(self, y, t):
        raise NotImplementedError("Each ODE must implement its own derivative!")

    def plot_results(self, t, y_rk4, y_euler, ref_y):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        ax1.plot(t, ref_y[:, 0], 'k--', label='Reference', alpha=0.7, lw=3)
        ax1.plot(t, y_rk4[:, 0], 'b--', label='RK4')
        ax1.plot(t, y_euler[:, 0], 'r--', label='Euler')
        ax1.set_ylabel("Position")
        ax1.legend()

        error_rk4 = y_rk4[:, 0] - ref_y[:, 0]
        error_euler = y_euler[:, 0] - ref_y[:, 0]
        
        ax2.plot(t, error_rk4, 'b-', label='RK4 Error')
        ax2.plot(t, error_euler, 'r-', label='Euler Error')
        ax2.set_yscale('log')
        ax2.set_ylabel("Error (Log Scale)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend()
        plt.title(f"{self.name} | Parameters: {self.params}")
        plt.tight_layout()
        plt.show()


    def get_benchmark_report(self, y_rk4, y_euler, ref_y):
        #MAE
        mae_rk4 = np.mean(np.abs(y_rk4[:, 0] - ref_y[:, 0]))
        mae_euler = np.mean(np.abs(y_euler[:, 0] - ref_y[:, 0]))
        
        #Scores are in Logarithmic MAE
        score_rk4 = -np.log10(mae_rk4 + 1e-16)
        score_euler = -np.log10(mae_euler + 1e-16)
        
        print(f"\n--- {self.name} Accuracy Report ---")
        print(f"Euler MAE: {mae_euler:.6f} (Score: {score_euler:.2f})")
        print(f"RK4   MAE: {mae_rk4:.6f} (Score: {score_rk4:.2f})")
        print(f"Conclusion: RK4 is {mae_euler/mae_rk4:.1f}x more accurate than Euler.")
        

class DampedOscillator(BaseODE):
    def __init__(self, init_cond, params, name="Damped Harmonic Oscillator"):
        super().__init__(init_cond, params, name)

    def derivs(self, y, t):
        m, c, k = self.params
        x, v = y
        return np.array([v, -c/m*v - k/m*x])
        

class VanderPol(BaseODE):
    def __init__(self, init_cond, params, name="Van der Pol Oscillator"):
        super().__init__(init_cond, params, name)

    def derivs(self, y, t):
        mu, = self.params
        y1, y2 = y
        return np.array([y2, mu*(1-y1**2)*y2 - y1])