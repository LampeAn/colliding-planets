import numpy as np
import scipy as sp

##todo remove dividing by zero

class Parameters:
    # Initialize with all parameters given.
    def __init__(self, dimension = 1, gamma = 1 ,initial_state = np.ndarray ,integration_time = 1, t_eval = np.array([1]), eta = 1.2, alpha_pi = 1, beta_pi = 1, rho_min = 1, h = None):
        self.dimension = dimension
        self.initial_state = initial_state

        ## total integration time
        self.integration_time = integration_time
        ## array of times to evaluate during intergration
        self.t_eval = t_eval

        ## parameters of matter
        self.gamma = gamma

        ## parameters for W
        ##eta should be in [1.2,1.5]
        self.eta = eta

        self.alpha_pi = alpha_pi
        self.beta_pi = beta_pi

        # Parameters have to be calculated using full state vector information this is done in the solver
        if h is None:
            self.h = self.calculate_h(rho_min)
        else:
            self.h = h
        print(self.h)
        self.a = self.calculate_a()

    @classmethod
    def from_e(cls, dimension = 1, gamma = 1 ,initial_state = np.ndarray, integration_time = 1, t_eval = np.array([1]), eta = 1.2, alpha_pi = 1, beta_pi = 1, rho_min = 1, h = None):
        ## initial state of form x,y,z,vx,vy,vz,m,e
        return cls(dimension = dimension,gamma = gamma ,initial_state = initial_state ,integration_time = integration_time,t_eval = t_eval,eta = eta ,alpha_pi = alpha_pi,beta_pi = beta_pi,rho_min = rho_min, h = h)
    
    @classmethod
    def from_p(cls, dimension = 1, gamma = 1 ,initial_state = np.ndarray, integration_time = 1, t_eval = np.array([1]), eta = 1.2, alpha_pi = 1, beta_pi = 1, rho_min = 1, h = None):
        ## initial state of form x,y,z,vx,vy,vz,m,rho,p
        ## calculate e from initial stare
        rho = initial_state[:,7]
        p = initial_state[:,8]
        e = p / ((gamma - 1) * rho)

        ## bring initial state into correct format
        n = len(initial_state[:,0])
        initial_state_new = np.zeros(shape=(n,8))
        initial_state_new[:,0:7] = initial_state[:,0:7]
        initial_state_new[:,7] = e

        return cls(dimension = dimension,gamma = gamma ,initial_state = initial_state_new ,integration_time = integration_time,t_eval = t_eval,eta = eta ,alpha_pi = alpha_pi,beta_pi = beta_pi,rho_min = rho_min, h = h)

    def calculate_h(self,rho_min):
        m_max = np.max(self.initial_state[:,6])
        h = self.eta * (m_max/rho_min)**(1/self.dimension)
        self.h = h
        return h
    
    def calculate_a(self):
        a = 0.0
        if self.dimension == 1:
            a = 1/self.h
        elif self.dimension == 2:
            a = 15/(7*np.pi * self.h**2)
        elif self.dimension == 3:
            a = 3/(2*np.pi * self.h**3)
        else:
            raise Exception
        self.a = a
        return a

class Solver:
    def __init__(self,parameters,use_artificial_viscosity : bool, use_gravity : bool):
        self.parameters = parameters
        self.use_artificial_viscosity = use_artificial_viscosity
        self.use_gravity = use_gravity
        return
    
    ## functions to calculate helping quantities
    def calculate_c(self,e):
        gamma = self.parameters.gamma
        return np.sqrt((gamma-1) * e)
    
    @classmethod
    def calculate_e(self,p,rho):
        gamma = self.parameters.gamma
        e = p / ((gamma - 1) * rho)
        return e

    def calculate_Pi_ij(self,v_ijk,x_ijk,rho,e):
        alpha_pi = self.parameters.alpha_pi
        beta_pi = self.parameters.beta_pi
        h = self.parameters.h

        rho_bar_ij = 0.5 * (rho[:,np.newaxis] + rho[np.newaxis,:])

        c = self.calculate_c(e)
        c_bar_ij = 0.5 * (c[:,np.newaxis] + c[np.newaxis,:])

        ## calculate v_ijk * v_ijk (scalar product of every vector combination) 
        dot_product = np.sum(v_ijk * x_ijk,axis = 2)

        h_ij = np.ones_like(v_ijk[:,:,0]) * h
        phi_ij = 0.1 * h_ij
        Phi_ij = h_ij * dot_product / (np.sum(x_ijk * x_ijk,axis=2) + phi_ij**2)
        
        Pi_ij = np.zeros_like(v_ijk[:,:,0])
        ## only values different form 0 if dot_product < 0
        mask = dot_product < 0
        Pi_ij[mask] = ((- alpha_pi * c_bar_ij * Phi_ij + beta_pi * Phi_ij**2) / rho_bar_ij)[mask].reshape(-1)

        return Pi_ij

    def calculate_x_ijk(self,state_vector):
        vec = state_vector[:,0:3]
        x_ijk = vec[:,np.newaxis,:] - vec[np.newaxis,:,:]
        return x_ijk

    def calculate_v_ijk(self,state_vector):
        vec = state_vector[:,3:6]
        v_ijk = vec[:,np.newaxis,:] - vec[np.newaxis,:,:]
        return v_ijk
    
    def calculate_r_ij(self,x_ijk):
        r_ij = np.sqrt(np.sum(x_ijk * x_ijk,axis=2))
        return r_ij

    def calculate_W_ij(self,r_ij):
        h = self.parameters.h
        a = self.parameters.a

        R_ij = r_ij/h
        W_ij = np.zeros_like(r_ij)

        case1_index = np.where((R_ij >=0) & (R_ij < 1))
        case2_index = np.where((R_ij >=1) & (R_ij <= 2))

        W_ij[case1_index] =  2/3 - R_ij[case1_index]**2 + 0.5 * R_ij[case1_index]**3
        W_ij[case2_index] = 1/6 * (2 - R_ij[case2_index])**3

        return a * W_ij

    def calculate_dW_ijk(self,r_ij,x_ijk):
        h = self.parameters.h
        a = self.parameters.a
        n = len(r_ij[:,0])

        R_ij = r_ij/h
        dW_ijk = np.zeros_like(x_ijk)
        #diagonal_mask = np.eye(n,dtype=bool)
        case1_index = ((R_ij >= 0) & (R_ij < 1))# & ~diagonal_mask
        case2_index = ((R_ij >=1) & (R_ij <= 2))# & ~diagonal_mask

        #dW_ijk[case1_index] = ((-2 + 3/2 * R_ij[:,:,np.newaxis])/h**2 * (x_ijk))[case1_index]
        #dW_ijk[case2_index] = (-1/2 * (2 - R_ij[:,:,np.newaxis])**2/(h**2 * R_ij[:,:,np.newaxis]) * x_ijk )[case2_index]
        dW_ijk[case1_index] = ((-2 + 3/2 * R_ij[:,:,np.newaxis][case1_index])/h**2 * (x_ijk[case1_index]))
        dW_ijk[case2_index] = (-1/2 * (2 - R_ij[:,:,np.newaxis][case2_index])**2/(h**2 * R_ij[:,:,np.newaxis][case2_index]) * x_ijk[case2_index] )

        return a * dW_ijk
    
    def calculate_dPhi_ij(self,r_ij):
        h = self.parameters.h
        #n = len(r_ij[:,0])
        R_ij = r_ij / h

        #diagonal_mask = np.eye(n,dtype=bool)

        dPhi_ij = np.zeros_like(r_ij)
        dPhi_ij = 1/h**2 * (4/3*R_ij - 6/5*R_ij**3 + 1/2*R_ij**4)

        case1_index = ((R_ij >= 1) & (R_ij < 2)) #& ~diagonal_mask
        case2_index = (R_ij >=2) #& ~diagonal_mask

        dPhi_ij[case1_index] = 1/h**2 * (8/3*R_ij[case1_index] - 3*R_ij[case1_index]**2 + 6/5*R_ij[case1_index]**3 -1/6*R_ij[case1_index]**4 -1/(15*R_ij[case1_index]**2))
        dPhi_ij[case2_index] = 1/r_ij[case2_index]**2

        return dPhi_ij
    
    ## calculates rho
    def calculate_rho(self,mass_j, W_ij):
        rho = np.einsum("j,ij->i",mass_j,W_ij)
        return rho

    ## calculates dv
    def calculate_dv(self,r_ij,m,p,rho,dW_ijk,Pi_ij,x_ijk):
        m_j = m[np.newaxis,:,np.newaxis]
        p_i = p[np.newaxis,:,np.newaxis]
        p_j = p[:,np.newaxis,np.newaxis]
        rho_i = rho[np.newaxis,:,np.newaxis]
        rho_j = rho[:,np.newaxis,np.newaxis]
        Pi_ij = Pi_ij.copy()[:,:,np.newaxis]

        dv = - np.sum(m_j * (p_i/rho_i**2 + p_j/rho_j**2 + Pi_ij) * dW_ijk,axis=1)
        
        ## handle gravity
        dv_g = np.zeros_like(dv)
        G = sp.constants.G
        if self.use_gravity:
            d_Phi_ij = self.calculate_dPhi_ij(r_ij=r_ij)[:,:,np.newaxis]
            r_ij = r_ij[:,:,np.newaxis]

            r_ij_save = np.where(r_ij > 1e-6, r_ij, 1)
            e_ijk = x_ijk/r_ij_save
            dv_g = - G * (m_j * d_Phi_ij) * e_ijk 
            dv_g = np.sum(dv_g,axis=1)
        return dv + dv_g
    
    ## calcualtes de
    def calculate_de(self,m,p,rho,dW_ijk,v_ijk,Pi_ij):
        m_j = m[:,np.newaxis]
        p_i = p[np.newaxis,:]
        p_j = p[:,np.newaxis]
        rho_i = rho[np.newaxis,:]
        rho_j = rho[:,np.newaxis]
        
        return 0.5 * np.sum(m_j * (p_i/rho_i**2 + p_j/rho_j**2 + Pi_ij) * np.sum(v_ijk * dW_ijk,axis = 2),axis=1)
    
    ## calcualte dx
    def calculate_dx(self,state_vector):
        return state_vector[:,3:6]

    def calculate_p(self,rho,e):
        gamma = self.parameters.gamma
        return (gamma - 1) * rho * e
    
    def calculate_full_state_vector(self,state_vector):
        '''
        Takes a state vector of shape (number_particles,8) with second axis [x,y,z,vx,vy,vz,m,e]
        
        It calculates rho and p for each particle and return an extended state_vector with shape (number particles,8) with second axis [x,y,z,vx,vy,vz,m,e,rho,p]
        '''
        x_ijk = self.calculate_x_ijk(state_vector=state_vector)
        r_ij = self.calculate_r_ij(x_ijk = x_ijk)
        W_ij = self.calculate_W_ij(r_ij=r_ij)
        rho = self.calculate_rho(W_ij=W_ij,mass_j=state_vector[:,6])
        p = self.calculate_p(rho=rho,e=state_vector[:,7])

        state_vector = np.concatenate((state_vector,rho[:,np.newaxis]),axis=1)
        state_vector = np.concatenate((state_vector,p[:,np.newaxis]),axis=1)

        return state_vector

    def calculate_derivative(self,t,state_vector):
        ## state vector [0 x,1 y,2 z,3 vx,4 vy,5 vz,6 m,7 e,8 rho,9 p]
        ##reshape state vector
        state_vector = state_vector.reshape((len(state_vector)//8,8))
        ## set internal energy to zero if negative
        state_vector[np.where(state_vector[:,7]<0),7] = 0

        derivative = np.zeros_like(state_vector)
        m = state_vector[:,6]
        e = state_vector[:,7]
        x_ijk = self.calculate_x_ijk(state_vector=state_vector)
        r_ij = self.calculate_r_ij(x_ijk = x_ijk)
        v_ijk = self.calculate_v_ijk(state_vector=state_vector)
        W_ij = self.calculate_W_ij(r_ij=r_ij)
        dW_ijk = self.calculate_dW_ijk(r_ij=r_ij,x_ijk=x_ijk)
        rho = self.calculate_rho(mass_j=m,W_ij=W_ij)
        p = self.calculate_p(e=e,rho=rho)

        ## calcualte artificial viscosity if desired
        Pi_ij = np.zeros_like(r_ij)
        if self.use_artificial_viscosity:
            Pi_ij = self.calculate_Pi_ij(v_ijk=v_ijk,x_ijk=x_ijk,rho=rho,e=e)

        ## calcualte differentials
        dx = self.calculate_dx(state_vector)
        dv = self.calculate_dv(dW_ijk=dW_ijk,m=m,p=p,rho=rho,r_ij=r_ij,Pi_ij=Pi_ij,x_ijk=x_ijk)
        de = self.calculate_de(rho=rho,p=p,m=m,dW_ijk=dW_ijk,v_ijk=v_ijk,Pi_ij=Pi_ij)
        dm = 0

        ## set differential of unused dimensions to zero to restric motion to specified dimension
        dimension = self.parameters.dimension
        dv[:,dimension:3] = 0
        ##restrict to 1D
        dx[:,dimension:3] = 0

        # populate the differential vector
        derivative[:,0:3] = dx
        derivative[:,3:6] = dv
        derivative[:,6] = dm
        derivative[:,7] = de

        return derivative.flatten()
    
    def integrate(self,initial_state,time,t_eval):
        res = sp.integrate.solve_ivp(self.calculate_derivative,(0,time),y0=initial_state,t_eval=t_eval)
        return res
    
    def run(self,filename):
        initial_state = self.parameters.initial_state
        integration_time = self.parameters.integration_time
        t_eval = self.parameters.t_eval
        ## integrate the problem to timeevolve
        res = self.integrate(initial_state = initial_state.flatten(),time = integration_time,t_eval = t_eval)

        ## calculate full statevectors from result
        data = np.zeros(shape = (len(res.y[:,0])//8 * 10, len(res.y[0,:])))
        for i in range(len(t_eval)):
            tmp_state_vector = np.reshape(res.y[:,i],(len(res.y[:,i])//8,8))
            data[:,i] = self.calculate_full_state_vector(state_vector = tmp_state_vector).flatten()

        ## save result to .csv file
        Solver.save_to_file(filename,res.t,data.T)

    @classmethod
    def save_to_file(self,filename,time,data):
        data = np.column_stack((time, data))
        np.savetxt(filename, data, delimiter=",", header="time, N * (x,y,z,vx,vy,vz,m,e,rho,p)")
        return
    
    @classmethod
    def read_from_file(self,filename):
        data = np.loadtxt(filename,skiprows=1,delimiter=",",ndmin=2)
        time = data[:,0]

        data = data[:,1:]
        data = data.reshape(len(time),-1,10)

        return time,data

class InitialState:
    def __init__(self,initial_state):
        self.initial_state = initial_state
        pass
    
    @classmethod
    def load_unstable_planet(self,filename):
        data = np.array(np.loadtxt("./planets/" + filename,skiprows=0,delimiter=" ",ndmin=2))
        return data
    
    @classmethod
    def load_stable_planet(cls,filename):
        data = np.loadtxt("./planets/" + filename,skiprows=1,delimiter=",",ndmin=2)
        time = data[:,0]
        data = data[:,1:]
        data = data.reshape(len(time),-1,10)
        return data[-1,:,:]
    
    @classmethod
    def load_sod_shock_tube(cls):
        #generate state vector
        x = np.concatenate((np.linspace(-0.6,-0.001875,320),np.linspace(0,0.6,81)))
        y = np.zeros_like(x)
        z = y.copy()

        vx = np.zeros_like(x)
        vy = vx.copy()
        vz = vx.copy()

        m = np.ones_like(x) * 0.001875
        e = np.concatenate((np.ones(320) * 2.5, 1.795 * np.ones(81)))

        data = np.c_[x,y,z,vx,vy,vz,m,e]
        return data 
    
    def calculate_center(self,m,x):
        return np.sum(m[:,np.newaxis] * x, axis = 0) / np.sum(m)
    
    def move_center_to_origin(self):
        m = self.initial_state[:,6]
        x = self.initial_state[:,0:3]
        R = self.calculate_center(m,x)
        self.move_center(-R)
        return self#self.initial_state
    
    def move_center_from_origin(self,R):
        self.move_center_to_origin()
        self.move_center(R)
        return self#self.initial_state
    
    def move_center(self,R):
        self.initial_state[:,0:3] = self.initial_state[:,0:3] + R
        return self#self.initial_state
    
    def add_boost(self,v):
        self.initial_state[:,3:6] = self.initial_state[:,3:6] + v
        return self
    
    def add_rotation(self,omega):
        r = self.initial_state[:,0:3]
        v_omega = np.cross(omega,r)

        self.initial_state[:,3:6] = self.initial_state[:,3:6] + v_omega
        return self
    
    def concatenate_initial_state(self,initial_state_tmp):
        self.initial_state = np.concatenate((self.initial_state,initial_state_tmp.get_initial_state()))
        return self
    
    def get_initial_state(self):
        return self.initial_state