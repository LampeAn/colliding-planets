import solver as sl
import numpy as np
from pathlib import Path
def main():
    planet_files = ["Planet300","Planet600","Planet1200","Planet2400"]

    for filename in planet_files:
        print("Starting to calculate stable form of planet: " + filename)
        initial_state = sl.InitialState.load_unstable_planet(filename + ".dat")
        #rho_min = np.min(initial_state[:,7])
        h = 1.2 * (5.873e24 / 245)**(1/3)
        parameters = sl.Parameters.from_p(dimension = 3, gamma = 1.4, initial_state = initial_state, integration_time = 500_000, t_eval = np.array([500_000]), eta = 1.2, alpha_pi = 1, beta_pi = 1, h = h)
        solver = sl.Solver(parameters=parameters,use_artificial_viscosity=True,use_gravity= True)
        solver.run(filename="./planets/" + filename + "_stable.dat")

if __name__ == "__main__":
    main()
