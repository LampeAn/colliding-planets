import solver as sl
import numpy as np
from pathlib import Path
def main():
    filename = "1200_1200_frontal_rotation"
    planet = "Planet1200_stable"

    print("Initializing Simulation")
    planet1 = sl.InitialState(sl.InitialState.load_stable_planet(planet + ".dat")).add_rotation([0,0,2 * np.pi / (24 * 3600)]).move_center_from_origin(R=[-2e9,0,0]).add_boost(v=[2e4,0,0])
    planet2 = sl.InitialState(sl.InitialState.load_stable_planet(planet + ".dat")).add_rotation([0,0,- 2 * np.pi / (24 * 3600)]).move_center_from_origin(R=[2e9,0,0]).add_boost(v=[-2e4,0,0])
    initial_state = planet1.concatenate_initial_state(planet2).get_initial_state()

    rho_min = np.mean(initial_state[:,8])
    parameters = sl.Parameters.from_e(dimension = 3, gamma = 1.4, initial_state = initial_state[:,:8], integration_time = 500_000, t_eval = np.linspace(0,500_000,200), eta = 1.2, alpha_pi = 1, beta_pi = 1, rho_min = rho_min)
    print("Start Simulation")
    solver = sl.Solver(parameters=parameters,use_artificial_viscosity=True,use_gravity= True)
    solver.run(filename = f"./simulation/{filename}.dat")

if __name__ == "__main__":
    main()
