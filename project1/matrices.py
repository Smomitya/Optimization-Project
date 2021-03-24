import numpy as np
import scipy
import scipy.sparse
import scipy.stats as ss

def make_adjacency_mat(n, p = 0.5, dtype = np.float64, column_stoch = False):
    assert n < 2 ** 20
    vals = np.random.choice(np.array([0, 1], dtype = dtype), size = n * (n - 1) // 2, p = [1-p, p])
#     print(vals)
    A = np.zeros((n,n), dtype = dtype)
    xi, yi = np.triu_indices(n,k = 1)
    assert len(xi) == n * (n - 1) // 2
    A[xi,yi] = vals
    A[yi,xi] = vals
    if column_stoch:
        return A / A.sum(axis = 0)
    else:
        return A

def make_sparse_adj_mat(n, p = 0.5, dtype = np.float64, column_stoch = False, verbose = False):
    num_1s = ss.binom.rvs(np.arange(1, n)[::-1], p) #number of ones in each row slice[i:n]
    
    if all(num_1s == np.zeros(n - 1)):
        num_1s[np.random.randint(0, n - 1)] = 1
        print('All zeros! Add 1.')
    if verbose:
        print(num_1s)
    xi = np.concatenate([num_1s[i]*[i] for i in range(n - 1) if num_1s[i] > 0]) #raw indices
    yi = np.concatenate([np.random.choice(np.arange(i+1, n), size = num_1s[i], replace = False) for i in range(n - 1) if num_1s[i] > 0]) #column indices
    assert len(xi) == len(yi)
    A = scipy.sparse.csr_matrix((np.ones(len(xi) * 2), (np.concatenate([xi, yi]), np.concatenate([yi, xi]))), shape = (n, n))
    if column_stoch:
        A.data /= np.array(np.maximum(1, A.sum(axis = 0)))[0,A.indices]
#         A.data /= np.take(A.sum(axis = 0), A.indices)
        return A
    else:
        return A

def get_sparse_size(matrix, b = 1):
    # get size of a sparse matrix in MiB
    return (matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes) / (1024.)**b