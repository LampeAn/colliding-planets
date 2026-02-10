import solver as sl
import numpy as np
from pathlib import Path
def main():
    filename = "sod_shock_tube"

    gamma = 1.4
    initial_state = sl.InitialState(sl.InitialState.load_sod_shock_tube()).get_initial_state()

    print("Initializing Simulation")

    h = 1.2 * (0.001875/ 0.25)**(1/3)
    parameters = sl.Parameters.from_e(dimension = 1, gamma = 1.4, initial_state = initial_state, integration_time = 0.2 , t_eval = np.linspace(0,0.2,2), eta = 1.2, alpha_pi = 1, beta_pi = 1, h = h)#rho_min = rho_min)# rho_min = rho_min)
    print("Start Simulation")
    solver = sl.Solver(parameters=parameters,use_artificial_viscosity=True,use_gravity= True)
    solver.run(filename = f"./simulation/{filename}.dat")

if __name__ == "__main__":
    main()
