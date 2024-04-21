
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
krylov = 'fgmres'
keep = False
strength=('classical', {'theta': 0.25})
splitting = ('RS', {'second_pass': True})
interp ='one_point'
restriction = ('air', {'theta': 0.05, 'degree': 2})
prerelax  =('cf_block_jacobi', {'omega': 1.0, 'iterations': 1, 'withrho': False, 'f_iterations': 2, 'c_iterations': 1})
postrelax = ('fc_block_jacobi', {'omega': 1.0, 'iterations': 1,'withrho': False, 'f_iterations': 2,'c_iterations': 1})
filter_entries = (True, 1e-4)
maxiter = 1000



# Load the problem matrix and filter the small entries
data = sio.loadmat('MMCP.mat')
A=data['A']
A = A.tocsr()
A.data[np.abs(A.data) < 1e-16] = 0
A.eliminate_zeros()
A=A.tobsr(blocksize=(16,16))
 
        
# Random initial guess and provided right hand side (Solving Ax = b)
x0 = np.random.rand(A.shape[0])
vec = np.loadtxt('vec.csv')
b = np.array(vec)

        
# Create a multilevel solver using approximate ideal restriction (AIR) AMG

ml = air_solver(A,strength=strength,CF=splitting,interpolation=interp,restrict=restriction,
                presmoother =prerelax,postsmoother =postrelax,filter_operator = filter_entries,
                max_levels=maximum_levels,keep=keep,max_coarse=maximum_coarse,coarse_solver=coarse_solver)



# Solve Ax=b with Krylov Method (AMG as a preconditioner to Krylov Method)

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
plt.grid(color='0.95')
plt.show()
