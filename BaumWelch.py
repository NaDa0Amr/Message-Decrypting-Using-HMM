import numpy as np

def log_sum_exp(log_probs):
    """
    Numerically stable log-sum-exp.
    Computes log(sum(exp(log_probs))) without underflow.
    """
    max_val = np.max(log_probs)
    return max_val + np.log(np.sum(np.exp(log_probs - max_val)))


def forward(obs, states, start_p, trans_p, emit_p):
    """
    Computes forward probabilities (alpha) in log space.
    Vectorized version for better performance.
    """
    T = len(obs)
    N = len(states)
    eps = 1e-100

    log_start = np.log(start_p + eps)
    log_trans = np.log(trans_p + eps)
    log_emit = np.log(emit_p + eps)

    alpha = np.full((T, N), -np.inf)
    alpha[0] = log_start + log_emit[:, obs[0]]

    for t in range(1, T):
        # Vectorized: compute all states at once
        for s in range(N):
            alpha[t, s] = log_sum_exp(alpha[t-1] + log_trans[:, s]) + log_emit[s, obs[t]]

    return alpha


def backward(obs, states, start_p, trans_p, emit_p):
    """
    Computes backward probabilities (beta) in log space.

    Parameters:
    - obs: observation sequence (list of indices)
    - states: list of state indices (usually range(N))
    - start_p: initial state probabilities (pi) - not used in backward but kept for consistency
    - trans_p: transition matrix (A)
    - emit_p: emission matrix (B)

    Returns:
    - beta: backward probabilities in log space, shape (T, N)
    """
    T = len(obs)
    N = len(states)
    eps = 1e-100

    log_trans = np.log(trans_p + eps)
    log_emit = np.log(emit_p + eps)

    # Initialize beta matrix with -inf
    beta = np.full((T, N), -np.inf)

    # Initialization: beta[T-1] = log(1) = 0 for all states
    beta[T-1] = 0.0

    # Induction: work backwards from T-2 to 0
    for t in range(T-2, -1, -1):
        # Vectorized: compute all states at once
        for s in range(N):
            # log P(future observations | current state s)
            log_prob_next_states = log_trans[s, :] + log_emit[:, obs[t+1]] + beta[t+1]
            beta[t, s] = log_sum_exp(log_prob_next_states)

    return beta


def compute_gamma(forward_probs, backward_probs):
    """
    Computes gamma (probability of being in state i at time t).
    gamma[t][s] = P(state=s at time t | all observations)

    Formula: gamma[t][s] = (alpha[t][s] * beta[t][s]) / sum_over_states(alpha[t][s] * beta[t][s])
    """
    T = len(forward_probs)
    N = len(forward_probs[0])
    gamma = np.zeros((T, N))

    # Computing gamma for each time step
    for t in range(T):
        # Combine forward and backward probabilities (in log space)
        log_prob = forward_probs[t] + backward_probs[t]  # log(alpha * beta)

        # Normalize using log-sum-exp
        log_norm_factor = log_sum_exp(log_prob)

        # Compute gamma for each state
        for s in range(N):
            gamma[t, s] = np.exp(log_prob[s] - log_norm_factor)

    return gamma


def compute_xi(obs, states, start_p, trans_p, emit_p, alpha, beta):
    """
    Computes xi (probability of transition from state i to state j at time t).
    xi[t][i][j] = P(state[t]=i, state[t+1]=j | all observations)

    Parameters:
    - obs: observation sequence
    - states: list of state indices
    - start_p, trans_p, emit_p: HMM parameters (pi, A, B)
    - alpha: forward probabilities (log space)
    - beta: backward probabilities (log space)

    Returns:
    - xi: transition probabilities, shape (T-1, N, N)
    """
    T = len(obs)
    N = len(states)
    eps = 1e-100

    log_trans = np.log(trans_p + eps)
    log_emit = np.log(emit_p + eps)

    xi = np.zeros((T-1, N, N))

    # For each time step (except the last one)
    for t in range(T-1):
        # Compute unnormalized log probabilities for all state pairs (i, j)
        # xi[t][i][j] ~ alpha[t][i] * A[i][j] * B[j][obs[t+1]] * beta[t+1][j]
        # In log space: log(alpha) + log(A) + log(B) + log(beta)

        # Broadcast to compute all (i,j) pairs at once
        log_xi = (alpha[t][:, np.newaxis] +           # alpha[t][i] for all i
                  log_trans +                          # A[i][j] for all i,j
                  log_emit[:, obs[t+1]] +              # B[j][obs[t+1]] for all j
                  beta[t+1])                           # beta[t+1][j] for all j

        # Normalize using log-sum-exp
        log_norm_factor = log_sum_exp(log_xi.flatten())  # Sum over all (i,j) pairs

        # Convert from log space to probabilities
        xi[t] = np.exp(log_xi - log_norm_factor)

    return xi


def Baum_Welch(obs, states, start_p, trans_p, emit_p, n_iter=30):
    """
    Baum-Welch that only updates emission matrix B (keeps A and pi fixed).
    Returns the learned emission matrix.
    """
    N = len(states)
    curr_emit_p = emit_p.copy()
    eps = 1e-10

    for iteration in range(n_iter):
        # E-STEP
        alpha = forward(obs, states, start_p, trans_p, curr_emit_p)
        beta = backward(obs, states, start_p, trans_p, curr_emit_p)
        gamma = compute_gamma(alpha, beta)

        # M-STEP: Update Emission Matrix (B) only
        new_emit_p = np.zeros_like(curr_emit_p)
        for t in range(len(obs)):
            new_emit_p[:, obs[t]] += gamma[t]

        new_emit_p += eps
        new_emit_p /= new_emit_p.sum(axis=1, keepdims=True)

        curr_emit_p = new_emit_p

    return curr_emit_p