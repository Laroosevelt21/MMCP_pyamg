import numpy as np
import scipy.sparse
import scipy.io as sio

# Load the CSV files
Mat = np.loadtxt('mat.csv')
Row = np.loadtxt('row.csv')
Col = np.loadtxt('col.csv')

# Convert into numpy arrays
Values = np.array(Mat)
Rows = np.array(Row,dtype=int)
Cols = np.array(Col,dtype=int)

# Create the COO matrix (in coordinate format)
A = scipy.sparse.coo_matrix((Values, (Rows, Cols)))

# Convert into CSR storage format
A = A.tocsr()

sio.savemat("MMCP.mat", {'A' : A})