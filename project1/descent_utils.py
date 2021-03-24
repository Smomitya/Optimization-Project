import numpy as np
import scipy
import scipy.sparse
from tqdm import tqdm
from functools import cache
from matrices import make_sparse_adj_mat
import matplotlib.pyplot as plt

def f(x, E, gamma):
    return 0.5 * np.sum((E @ x - x)**2) + 0.5 * gamma * (np.sum(x) - 1) ** 2

def coord_grad(x, k, E, gamma, verbose = False):
    '''
    x - vector of len N
    E - NxN sparse matrix
    gamma - float
    k - list of coordinate group indices
    
    return array of updated coordinates
    '''
    u = np.ones(len(k)) * gamma * (np.sum(x) - 1)
    if verbose:
        print(len(k), x.shape, E.shape)
    diff = E @ x - x
    return u + E.T[k] @ diff - E[k] @ x + x[k]
        
def U(g, k, N):
    '''
    N - number of dimension for initial vector
    k - index list for a group indices
    g - gradient vector
    '''
    assert len(g) == len(k)
    u = np.zeros(N)
    u[k] = g
    return u

def fast_get_L(gr, E, gamma, ord = 2):
    '''
    E - NxN sparse matrix
    gamma - float
    gr - list (dict) of index lists
    '''
    res = []
    A = E - scipy.sparse.eye(E.shape[0], format = 'csr')
    ATA = A.T @ A
    for k in gr:
        res.append(np.linalg.norm(ATA[k][:,k].toarray() + gamma * np.ones((len(k),len(k))), ord = ord))
    return res

def get_L(gr, E, gamma, ord = 2):
    '''
    E - NxN sparse matrix
    gamma - float
    gr - list (dict) of index lists
    '''
    res = []
    A = E - scipy.sparse.eye(E.shape[0], format = 'csr')
    for k in gr:
        Ak = (A.T[k] @ A[:, k]).toarray()
        res.append(np.linalg.norm(Ak + gamma * np.ones((len(k),len(k))), ord = ord))
    return res

def get_1d_L(E, gamma):
    '''
    1-D analog of get_L
    '''
    A = E.transpose()
    res = []
    for i in range(A.shape[0]):
        res.append(gamma + (A[i] @ A[i].T)[0,0] - 2 * A[i, i] + 1)
    return res

# def make_group(n, size):
#     group_size, part = divmod(n, size)
#     groups = []
#     group = []
#     for i in range(n + 1):
#         if len(group) < group_size + (1 if len(groups) < part else 0):
#             group.append(i)
# #             print(group)
#         else:
#             groups.append(group)
#             group = [i]
            
#     return groups

def make_group(n, size):
    '''
    n : range[0:n]
    size : size of a single group. n mod size  = 0!
    '''
    return np.split(np.arange(n), n // size)

def run_iters(x0, groups, E, gamma, L = None, eps = 1e-2, alpha = 1, max_iter = 100, deter = False, verbose = False):
    '''
    x0 : array of size N
    E : NxN matrix
    gamma : float parameter
    groups : list of lists, containing indices. E.g. [[1,2], [3,4]]
    deter : deterministic strategy of choosing group index or not
    alpha : distribution parameter for Liptshitz constants 
    '''
    if L is None:
        L = get_L(groups, E, gamma)
    L = np.array(L)
    
    if not deter:
        L_p = L ** alpha / np.sum(L ** alpha)
    list_f = []
    num_groups = len(groups)
    x = x0.copy()
    f = lambda x : 0.5 * np.sum((E @ x - x)**2) + 0.5 * gamma * (np.sum(x) - 1) ** 2
    
    @cache
    def cross_mat(idx):
        '''
        return E.T[k] @ E - E[k] - E.T[k]
        '''
        return E.T[groups[idx]] @ E - E[groups[idx]] - E.T[groups[idx]]
    
    def cached_grad(x, idx, verbose = False):
        '''
        x - vector of len N
        idx : int, index of a specific coord group 
        return list of updated coordinates
        '''
        u = np.ones(len(groups[idx])) * gamma * (np.sum(x) - 1)
#         if verbose:
#             print(len(groups[idx]), x.shape, E.shape)
        return cross_mat(idx) @ x + x[groups[idx]] + u
    
    for k in tqdm(range(max_iter)): #full iteration over all coordinates
        for gr_it in range(num_groups): #partial iteration over subgroup of coordinates 
            if deter:
    #             idx = k %  num_groups #cyclic strategy
                idx = gr_it
            else:
                idx = np.random.choice(np.arange(0, num_groups), p = L_p)

#             x[groups[idx]] = x[groups[idx]] - 1 / L[idx] * coord_grad(x, groups[idx], gamma, cross_mat, verbose = verbose)
            x[groups[idx]] = x[groups[idx]] - 1 / L[idx] * cached_grad(x, idx)
            
#         if np.sqrt(np.power(E @ x, 2).sum()) < eps * np.sqrt(np.power(x, 2).sum()):
#             break
            
        f_x = f(x)
        list_f.append(f_x)
        
    return list_f

# from tqdm import tqdm
def run_experiment(pow_n = 12, freq = 1, Ls = None, gamma = None, max_iter = 100, x0 = None, seed = 1, deter = False , alpha = 1, ignore_list = []):
    n = 2 ** pow_n
    p = 10 / n

    if gamma is None:
        gamma = 1 / np.sqrt(n)
        
    np.random.seed(seed)
    
    E_ = make_sparse_adj_mat(n, p = p, column_stoch=True)
    # list_k = [16, 32, 64, 128, 256, 512, 1024, 2048]
    domain = [i for i in range(0, pow_n, freq)]
    if domain[-1] != pow_n:
        domain += [pow_n]
    domain = np.array(domain)
    if Ls is None:
        Ls = {k : get_L(make_group(n, k), E_, gamma) for k in 2 ** domain if k not in ignore_list}
    print(Ls.keys())
    
    logs = {'pow_n': pow_n, 'n' : n, 'gamma' : gamma, 'L' : Ls, 'E' : E_, 'a' : alpha}
    logs['strategy'] = 'Cyclic' if deter else 'Random'
    logs['x0'] = 'e * 1/n' if x0 is None else 'random'
    
    if x0 is None:
        x0 = np.ones(n) / n
    
    dict_groups = {}
    for k in Ls.keys():
#         if k in ignore_list:
#             continue
        indices = make_group(n, k)

#         max_iter = 200
        np.random.seed(seed)
        dict_groups[k] = run_iters(x0, indices, E_, gamma, L = Ls[k], max_iter = max_iter, alpha = alpha, deter = deter)
    logs['loss'] = dict_groups
    
    return logs

def plot_results(log, figsize=(15, 9), end = 50, step = 1):
    plt.figure(figsize=figsize)
    plt.grid(color='lightgray',linestyle='--')
    title = r'$%s, \alpha = %d, x_0 = %s, n = %d$' % (log['strategy'], log['a'], log['x0'], log['n'])
    plt.title(title)
    plt.xlabel(r'$n$, iteration')
    plt.ylabel(r'$\log(f(x_n))$, loss')
    for i, k in enumerate(log['loss'].keys()):
        if i % step == 0 or k == log['n']:
            plt.semilogy(log['loss'][k][:end], label=f'group size {k}' if k != log['n'] else 'full grad')
    plt.legend()
    x0 = 'r' if log['x0'] == 'random' else 'u'
    if x0 == 'u':
        plt.hlines(1e-11 if log['strategy'] == 'Random' else 1e-8, 0, 50, color = 'red', linestyle = '--')
    plt.savefig('./pics/%s.png' % (log['strategy'] + str(log['pow_n']) + x0 + str(log['a'])));