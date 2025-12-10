# -*- coding: utf-8 -*-
"""Viterbi decoding algorithm for HMM."""

import numpy as np

def log_sum_exp(log_probs):
    """
    Numerically stable log-sum-exp.
    Computes log(sum(exp(log_probs))) without underflow.
    """
    max_val = np.max(log_probs)
    return max_val + np.log(np.sum(np.exp(log_probs - max_val)))

def Viterbi_decoding(obs, states, start_p, trans_p, emit_p, verbose=False):
    """
    Viterbi algorithm - vectorized for better performance.
    """
    T = len(obs)
    N = len(states)
    eps = 1e-100

    log_start = np.log(start_p + eps)
    log_trans = np.log(trans_p + eps)
    log_emit = np.log(emit_p + eps)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    delta[0] = log_start + log_emit[:, obs[0]]

    # Recursion (vectorized)
    for t in range(1, T):
        # Compute all transitions at once: (N, N) matrix
        # delta[t-1][:, newaxis] broadcasts to (N, 1)
        # log_trans is (N, N)
        # Result: (N, N) where [i, j] = delta[t-1][i] + log_trans[i][j]
        temp = delta[t-1][:, np.newaxis] + log_trans

        # Find best previous state for each current state
        psi[t] = np.argmax(temp, axis=0)

        # Store max values and add emission probability
        delta[t] = np.max(temp, axis=0) + log_emit[:, obs[t]]

    # Backtracking
    path = np.zeros(T, dtype=int)
    path[T-1] = np.argmax(delta[T-1])

    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    if verbose:
        print(f"✅ Viterbi decoding complete")
        print(f"   Best path log prob: {delta[T-1, path[T-1]]:.4f}")

    return path