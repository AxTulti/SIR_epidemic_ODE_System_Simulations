import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- 1. Define the SIR model differential equations ---
def deriv(y, t, N, beta, gamma):
    """
    Defines the derivatives for the SIR model.
    S, I, R = y
    dS/dt, dI/dt, dR/dt = derivatives
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# --- 2. Set Parameters and Initial Conditions ---
N = 1000          # Total population
I0 = 1            # Initial number of infected individuals
R0 = 0            # Initial number of recovered individuals
S0 = N - I0 - R0  # Initial number of susceptible individuals

beta = 0.3        # Contact rate (transmission rate)
gamma = 0.1       # Mean recovery rate (1/days)

# A grid of time points (in days)
t = np.linspace(0, 100, 1000) # Simulate for 100 days with 1000 points

# Initial conditions vector
y0 = S0, I0, R0

# --- 3. Integrate the SIR equations over the time grid ---
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T # Transpose the results to get arrays for S, I, R over time

# --- 4. Calculate R_eff over time ---
# Reff(t) = R0 * S(t) / N, where R0 = beta / gamma
R0_initial = beta / gamma
R_eff_over_time = R0_initial * S / N

# --- 5. Plot the R_eff against time ---
plt.figure(figsize=(8, 5))
plt.plot(t, R_eff_over_time, 'r-', label='$R_{eff}(t)$')
plt.axhline(1, color='gray', linestyle='--', linewidth=2, label='Threshold ($R_{eff}=1$)') # Add threshold line
plt.xlabel('Time (days)')
plt.ylabel('Effective Reproduction Number ($R_{eff}$)')
plt.title('Effective Reproduction Number Over Time in a Basic SIR Model')
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, max(R_eff_over_time) + 0.5)
plt.show()