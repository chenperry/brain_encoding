"""
This file is based on the implementation from: https://github.com/mmxgn/smooth-convex-kl-nmf
The objective function was modified from KL Divergence to MSE. 
"""
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except:
    from tqdm import tqdm as tqdm


def smooth(H):
    # Used as regularization for "smoothing" the resulting activations across columns
    return 0.5 * np.sum((H[:, :-1] - H[:, 1:]) ** 2)


def smooth_rows(H):
    # Same as above, but smooths the rows instead
    return 0.5 * np.sum((H[:, :-1] - H[:, 1:]) ** 2, axis=1)


def objfunc(V, W, H, beta=0.0):
    # MSE
    V_hat = np.dot(W, H)
    return np.sqrt(mean_squared_error(V, V_hat))
    


def update_h_given_w(V, W, H, beta=0.0, lamda=None):
    F = V.shape[0]
    N = V.shape[1]
    K = W.shape[1]

    if lamda is None:
        # Lamda can be precomputed once
        lamda = np.sum(np.abs(W), axis=0)
    lamda_col = lamda.reshape(-1, 1)
    V_hat = np.matmul(W, H)

    psi = H * np.matmul(W.T, V / V_hat)
    psi = psi.astype('float64')  # object-->float64

    if beta == 0:
        H_new = psi / wcolnorms
        return H_new
    else:
        a = np.zeros((K, N))
        b = np.zeros((K, N))

        # Edge cases
        b[:, 0] = lamda * (1 - beta * lamda * H[:, 1])
        a[:, 0] = a[:, N - 1] = beta * lamda ** 2
        a[:, 1:N - 1] = 2 * beta * lamda_col ** 2
        b[:, N - 1] = lamda * (1 - beta * lamda * H[:, N - 2])

        # Rest of cases

        Dh = (H[:, :-2] + H[:, 2:])
        b[:, 1:-1] = lamda_col * (1 - beta * lamda_col * Dh)

        H_new = (np.sqrt(b ** 2 + 4 * a * psi) - b) / (2 * a)
        return H_new


def update_w_given_h(V, W, H, beta=0.0):
    # Consider H fixed, then update towards a better W. Used in smoothNMF.
    # beta is the regularization penalizing weight.
    F = V.shape[0]
    N = V.shape[1]
    K = W.shape[1]
    vhat = np.matmul(W, H)
    phi = W * np.matmul(V / vhat, H.T)
    phi = phi.astype('float64')
    
    sigma_k = np.sum(H, axis=1).reshape(1, K)

    if beta == 0:

        W_new = phi / sigma_k
        return W_new
    else:
        a = np.zeros((F, K))
        b = np.zeros((F, K))
        for f in range(F):
            for k in range(K):
                s_k = 2 * smooth(H[k, :].reshape(1, -1))
                a[f, k] = beta * s_k
                b[f, k] = sigma_k[0, k] + beta * s_k * np.sum(W[np.arange(F) != f, k])

        W_new = (np.sqrt(b ** 2 + 4 * a * phi) - b) / (2 * a)
        return W_new


def update_l_given_h(V, L, H, beta=0.001):
    F = V.shape[0]
    N = V.shape[1]
    M = L.shape[0]
    K = L.shape[1]

    vhat = np.matmul(np.matmul(V, L), H)

    # The following computes the inner sum V^v_{kn} = \sum_{fn}{v_{fn}/\hat{v}_{fn}*f_{kn}
    VV = np.matmul(V / vhat, H.T)

    # Computes phi as phi_{mk}
    phi = L * np.matmul(V.T, VV)
    phi = phi.astype('float64')
    
    sigma_k = np.sum(H, axis=1).reshape(1, K)
    s_k = smooth_rows(H).reshape(1, -1)

    delta = np.sum(V, axis=0).reshape(-1, 1)
    a = beta * np.matmul(delta ** 2, s_k)
    DL = delta * L

    sumColsDL = np.sum(DL, 1).reshape(-1, 1)
    sumDL = np.sum(sumColsDL)
    sumDL = np.full(sumColsDL.shape, sumDL)

    sigma_k_delta = np.matmul(delta, sigma_k)

    b = np.zeros((M, K))
    b += sumDL.astype(float) - sumColsDL.astype(float)
    b *= np.array(beta).astype(float) * np.array(delta).astype(float)
    b += np.array(sigma_k_delta).astype(float)
    # print(np.any((b ** 2 + 4 * a * phi) < 0))
    L_new = (np.sqrt(b ** 2 + 4 * a.astype(float) * phi.astype(float)) - b) / (2 * a.astype(float))

    return L_new


def smoothConvexNMF(V, k, beta=0.001, max_iter=100, n_trials_init=10, init='random'):
    # Smooth and Convex NMF constraints the activations H to be smooth and "sparse"
    V = np.array(V)
    if init == 'random':
        # Initialize randomly after some trials
        best_cost = np.inf
        for n in range(n_trials_init):

            L = np.abs(np.random.randn(V.shape[1], k))
            H = np.abs(np.random.randn(k, V.shape[1]))

            W = np.matmul(V, L)
            cost = objfunc(V, W, H, beta)

            if cost < best_cost:
                Lh = L
                Hh = H
                best_cost = cost
    else:
        # If init is not 'random' then use predefined matrices
        Lh = init['L']
        Hh = init['H']

    costs = np.zeros((max_iter,))
    last_cost = np.inf

    for I in tqdm(range(max_iter)):
        cur_cost = objfunc(V, np.matmul(V, Lh), Hh, beta)

        # cost_diff = np.abs(cur_cost - last_cost)
        last_cost = cur_cost

        if I > 0:
            costs[I - 1] = last_cost

        Hh = update_h_given_w(V, np.matmul(V, Lh), Hh, beta)
        Lh = update_l_given_h(V, Lh, Hh, beta)

    return Lh, Hh, costs[:I]

def miniBatchSmoothConvexNMF(V, k, batch_size=5, epochs=1000, beta=0.001, tol=1e-8, sort=True, init='random'):
    best_cost = np.inf

    costs = []
    batchindices = np.array_split(np.arange(V.shape[1]), V.shape[1] / batch_size)

    if init == 'random':
        # Initialize H, L randomly
        H = np.abs(np.random.randn(k, V.shape[1]))
        L = np.abs(np.random.randn(V.shape[1], k))
    else:
        # If init is not 'random' then use predefined matrices
        H = init['H']
        L = init['L']

    for epoch in tqdm(range(epochs)):
        W = np.matmul(V, L)
        lamda = np.sum(np.abs(W), axis=0)


        for n, batchidx in enumerate(batchindices):
            # Update the activations for each batch
            if n > 0 and n < len(batchindices) - 1:
                H[:, batchidx] = update_h_given_w(V[:, min(batchidx) - 1:max(batchidx) + 2], W,
                                                  H[:, min(batchidx) - 1:max(batchidx) + 2], beta, lamda=lamda)[:, 1:-1]
            if n == 0:
                H[:, batchidx] = update_h_given_w(V[:, min(batchidx):max(batchidx) + 2], W,
                                                  H[:, min(batchidx):max(batchidx) + 2], beta, lamda=lamda)[:, :-1]
            if n == len(batchindices) - 1:
                H[:, batchidx] = update_h_given_w(V[:, min(batchidx) - 1:max(batchidx) + 1], W,
                                                  H[:, min(batchidx) - 1:max(batchidx) + 1], beta, lamda=lamda)[:, 1:]

        # Update the dictionary once per epoch
        L = update_l_given_h(V, L, H, beta)

        cur_cost = objfunc(V, np.matmul(V, L), H, beta)
        costs.append(cur_cost)
    return L, H, costs


def calculate_cnmf_r_squared(X, X_hat):
    """
    Calculate the coefficient of determination (R^2).

    Parameters:
    - X: Observed values
    - X_hat: Predicted values

    Returns:
    - R_squared: Coefficient of determination
    """
    # Calculate the mean of the observed values
    X = np.array(X)
    mean_X = np.mean(X)

    # Calculate the total sum of squares (TSS)
    TSS = np.sum((X - mean_X)**2)

    # Calculate the residual sum of squares (RSS)
    RSS = np.sum((X - X_hat)**2)

    # Calculate R^2 using the formula
    R_squared = 1 - (RSS / TSS)

    return R_squared


def smoothNMF(V, k, beta=0.0, tol=1e-8, max_iter=100, n_trials_init=10):
    # smoothNMF constraints the activations to be "smooth"

    # Initialize randomly after some trials
    best_cost = np.inf

    for n in range(n_trials_init):

        W = np.abs(np.random.randn(V.shape[0], k))
        H = np.abs(np.random.randn(k, V.shape[1]))
        cost = objfunc(V, W, H, beta)
        if cost < best_cost:
            Wh = W
            Hh = H
            best_cost = cost

    costs = np.zeros((max_iter,))
    last_cost = np.inf
    for I in tqdm(range(max_iter)):
        cur_cost = objfunc(V, Wh, Hh, beta)

        cost_diff = np.abs(cur_cost - last_cost)
        if cost_diff <= tol:
            break

        last_cost = cur_cost

        if I > 0:
            costs[I - 1] = last_cost

        Hh = update_h_given_w(V, Wh, Hh, beta)
        Wh = update_w_given_h(V, Wh, Hh, beta)

    return Wh, Hh, costs[:I]