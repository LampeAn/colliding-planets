import solver as sl
import numpy as np
from pathlib import Path
def main():
    planet_files = ["Planet300","Planet600","Planet1200","Planet2400"]

    for filename in planet_files:
        print("Starting to calculate stable form of planet: " + filename)
        initial_state = sl.InitialStatePreparer().load_unstable_planet(filename + ".dat")
        rho_min = np.min(initial_state[:,7])
        parameters = sl.Parameters.from_p(dimension = 3, gamma = 1.4, initial_state = initial_state, integration_time = 500_000, t_eval = np.linspace(0,500_000,200), eta = 1.2, alpha_pi = 1, beta_pi = 1, rho_min = rho_min)
        solver = sl.Solver(parameters=parameters,use_artificial_viscosity=True,use_gravity= True)
        solver.run(filename="./planets/" + filename + "_stable.dat")

if __name__ == "__main__":
    main()
