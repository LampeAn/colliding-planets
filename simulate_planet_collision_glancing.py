import solver as sl
import numpy as np
from pathlib import Path
def main():
    filename = "300_300_glancing1"
    planet = "Planet300_stable"

    print("Initializing Simulation")
    planet1 = sl.InitialState(sl.InitialState.load_stable_planet(planet + ".dat")).move_center_from_origin(R=[-2e9,0.9e8,0]).add_boost(v=[2e4,0,0])
    planet2 = sl.InitialState(sl.InitialState.load_stable_planet(planet + ".dat")).move_center_from_origin(R=[2e9,-0.9e8,0]).add_boost(v=[-2e4,0,0])
    initial_state = planet1.concatenate_initial_state(planet2).get_initial_state()

    h = 1.2 * (5.873e24 / 245)**(1/3)
    parameters = sl.Parameters.from_e(dimension = 3, gamma = 1.4, initial_state = initial_state[:,:8], integration_time = 500_000, t_eval = np.linspace(0,500_000,200), eta = 1.2, alpha_pi = 1, beta_pi = 1, h = h)#rho_min = rho_min)# rho_min = rho_min)
    print("Start Simulation")
    solver = sl.Solver(parameters=parameters,use_artificial_viscosity=True,use_gravity= True)
    solver.run(filename = f"./simulation/{filename}.dat")

if __name__ == "__main__":
    main()
