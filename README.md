# SIR Epidemic Modeling: Analytical Solutions and Agent-Based Simulations
 
Modeling disease spread in a closed population with the SIR (Susceptible–Infected–Recovered) model: analytical derivations, numerical ODE solutions in R, and 8 agent-based simulations in Python that reproduce the ODE behavior experimentally.
 
 
**Team project, Tecnológico de Monterrey (Team 5):** Axel Yael Rodríguez Romero, Ángel Nicolás Landa Tapia, Juan Diego Rodríguez Espinoza, Emilio Estrada Pérez.
**My contributions:** analytical derivations and the Python simulations.
 
## Publications
 
- 📄 [Part 1: SIR Model (RPubs)](https://rpubs.com/AxRodro/SIR_Model_1)
- 📄 [Part 2: SIR Model (RPubs)](https://rpubs.com/AxRodro/SIR_2)
- 📹 [Video of the simulations](https://drive.google.com/file/d/1LuTyB34fJdno6j2cBShnVwiP8teuee3J/view?usp=sharing)
## Key results
 
| Result | Analytical | Numerical |
|---|---|---|
| Day of peak infection (β = 1, γ = 0.1) | 18.0028 days | 18 days |
| Day when S = N/2 | 15.2886 days | 15 days |
| Herd immunity threshold (β = 0.6, γ = 0.1) | v = 5/6 vaccinated | Epidemic avoided at v = 5/6, occurs at v = 2/3 |
 
Agent-based simulations show that **population movement and clustering significantly accelerate the spread** of infection compared with static populations.
 
## Models
 
### 1. Constant population
 
```math
\frac{dS}{dt} = -\beta \frac{I}{N} S, \qquad
\frac{dI}{dt} = \beta \frac{I}{N} S - \gamma I, \qquad
\frac{dR}{dt} = \gamma I
```
 
where N = S + I + R, β is the infection rate and γ the recovery rate. The basic and effective reproduction numbers are
 
```math
R_0 = \frac{\beta}{\gamma}, \qquad R_{\text{eff}}(t) = R_0 \frac{S(t)}{N}
```
 
and an epidemic occurs when R₀ > 1.
 
**Peak infection time.** Solving for S in terms of R gives S = (N − 1)·e^(−βR / Nγ). Since I peaks when S = γN / β:
 
```math
t_{I_{\max}} = \int_{\gamma N/\beta}^{N-1} \frac{N}{\beta S \left[ N - S - \frac{\gamma N}{\beta}\left(\ln(N-1) - \ln S\right) \right]} \, dS
```
 
**Time until half the population has been infected:**
 
```math
t_{S = N/2} = \int_{0}^{\frac{N\gamma}{\beta}\left[\ln(2N-2) - \ln N\right]} \frac{dR}{\gamma \left( N - R - (N-1)\,e^{-\beta R/(N\gamma)} \right)}
```
 
**Parameter effects.** Increasing β brings the infection peak forward; increasing γ delays it and reduces it. If R₀ ≤ 1, no epidemic occurs.
 
### 2. Births and deaths
 
With birth rate b and death rate μ:
 
```math
\frac{dS}{dt} = bN - \beta \frac{I}{N} S - \mu S, \qquad
\frac{dI}{dt} = \beta \frac{I}{N} S - \gamma I - \mu I, \qquad
\frac{dR}{dt} = \gamma I - \mu R
```
 
When b = μ the population stays constant; when b < μ it declines over time (shown with μ = 1/10, b = 1/70).
 
### 3. Vaccination and herd immunity
 
Vaccinating an initial fraction v of the susceptible population (moving it to R) prevents the epidemic when
 
```math
v > \frac{\frac{\gamma}{\beta} - 1 + \frac{1}{N}}{-1 + \frac{1}{N}} \approx 1 - \frac{1}{R_0}
```
 
## Agent-based simulations
 
Eight configurations simulated in Python, combining:
 
- **Geometry:** square city vs. circular city
- **Distribution:** uniform vs. clustered population
- **Dynamics:** static individuals vs. moving individuals
Static simulations show abrupt state changes that only loosely follow the SIR curves. Moving populations produce clearly SIR-shaped dynamics, and clustered populations reach their peak and equilibrium much faster than unclustered ones.
 
## Repository structure
 
```
├── first_no_movement.py    # Static simulation: square city
├── second_no_movement.py   # Static simulation: circular city
├── third_no_movement.py    # Static simulation: clustered square city
├── forth_no_movement.py    # Static simulation: clustered circular city
├── 4fase/                  # Movement simulations
├── Act 1.Rmd / Act-1.html  # Part 1: constant population model (R)
├── Act 2.Rmd / Act-2.html  # Part 2: births, deaths and vaccination (R)
└── SIR_MODEL_FINAL.pdf     # Executive summary
```
 
## How to run
 
**Python simulations**
 
```bash
pip install numpy matplotlib pandas
python first_no_movement.py
```
 
Run any of the other scripts the same way, or the ones in `4fase/` for the movement simulations.
 
**R models:** open `Act 1.Rmd` or `Act 2.Rmd` in RStudio and install the required packages:
 
```r
install.packages(c("deSolve", "reshape2", "ggplot2"))
```
 
## Tools
 
- **Python:** NumPy, Matplotlib, Pandas, random (stochastic simulation)
- **R:** deSolve (ODE solving), reshape2, ggplot2
## Report
 
The full executive summary with derivations and figures: [SIR_MODEL_FINAL.pdf](SIR_MODEL_FINAL.pdf)
