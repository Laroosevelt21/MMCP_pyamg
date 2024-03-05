
# Import required packages
import pyamg
from pyamg.classical import air_solver
from pyamg.aggregation import energymin_cf_solver
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



# AMG Parameters
maximum_levels = 30         # Max levels in hierarchy
maximum_coarse = 30         # Max points allowed on coarse grid
tolerance = 1e-9      # Residual convergence tolerance
coarse_solver = 'splu'
krylov = 'gmres'
keep = False
strength=('classical', {'theta': 0.01})
splitting = ('RS', {'second_pass': True})
interp ='classical'
restriction = ('air', {'theta': 0.05, 'degree': 2})
prerelax  =('cf_jacobi', {'omega': 1.0, 'iterations': 1, 'withrho': True, 'f_iterations': 2, 'c_iterations': 1})
postrelax = ('fc_jacobi', {'omega': 1.0, 'iterations': 1,'withrho': True, 'f_iterations': 2,'c_iterations': 1})
#prepost = ('gauss_seidel_nr', {'sweep': 'symmetric', 'iterations': 1})
filter_entries = None
maxiter = 1000



# Load the problem matrix and filter the small entries
data = sio.loadmat('MMCP.mat')
A=data['A']
A = A.tocsr()
A.data[np.abs(A.data) < 1e-16] = 0
A.eliminate_zeros() 
        
# Random initial guess and zero right hand side (Solving Ax = 0)
x0 = np.random.rand(A.shape[0])
b = np.zeros((A.shape[0],))

        
# Create a multilevel solver using approximate ideal restriction (AIR) AMG

ml = air_solver(A,strength=strength,CF=splitting,interpolation=interp,restrict=restriction,
                presmoother =prerelax,postsmoother =postrelax,filter_operator = filter_entries,
                max_levels=maximum_levels,keep=keep,max_coarse=maximum_coarse,coarse_solver=coarse_solver)


# Create a multilevel solver using classical Ruge-Stuben AMG Solver

# ml = pyamg.ruge_stuben_solver(A,strength = strength,CF = splitting,interpolation = interp,
#                               presmoother = prerelax,postsmoother = postrelax,
#                               max_levels = maximum_levels,max_coarse = maximum_coarse,
#                               keep = keep,coarse_solver=coarse_solver)


# Solve Ax=b with GMRES (AMG as a preconditioner to GMRES)
resvec=[]
x = ml.solve(b, x0=x0, maxiter = maxiter, tol = tolerance ,accel=krylov, residuals=resvec)
factors_air = (resvec[-1] / resvec[0])**(1.0 / len(resvec))
Iterations = len(resvec)
complexity_air = ml.operator_complexity()
grid_complexity_air=ml.grid_complexity()
size_air = A.shape[0]
print(ml)
print(resvec)
        
        
# Print AMG detailed results
print("AMG solver results ")
print("{:^9s} | {:^9s}  | {:^9s} | {:^9s} | {:^9s} | {:^9s}".format(
    "n", "iter", "rho", "OpCx", "GdCx", "Work"))
print("-----------------------------------------------------------------------")

print("{:^9d} | {:^9d}  | {:^9.2g} | {:^9.2g} | {:^9.2g} | {:^9.2g}".format(
   size_air, Iterations, factors_air,complexity_air,grid_complexity_air,
    complexity_air / (-np.log10(factors_air))))

plt.semilogy(resvec/resvec[0], '-')
plt.xlabel('iterations')
plt.ylabel('normalized residual')
plt.show()
