# -*- coding: utf-8 -*-
"""Utility functions for HMM-based cipher breaking."""

import numpy as np
import random
import os
import nltk

# Download NLTK data (run once)
try:
    nltk.data.find('corpora/gutenberg')
except LookupError:
    nltk.download('gutenberg', quiet=True)

from nltk.corpus import gutenberg

# Model save path
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "hmm_model.npz")


def generate_training_corpus(n_samples=1000, min_length=50):
    """
    Generate training corpus from NLTK's Gutenberg corpus.
    Creates n_samples of long text segments.
    """
    # Get all text from Gutenberg corpus
    all_text = ""
    for fileid in gutenberg.fileids():
        all_text += gutenberg.raw(fileid) + " "
    
    # Clean and uppercase
    all_text = all_text.upper()
    
    # Keep only letters and spaces
    cleaned_text = ""
    for char in all_text:
        if char.isalpha() or char == ' ':
            cleaned_text += char
        else:
            cleaned_text += ' '  # Replace punctuation with space
    
    # Remove multiple spaces
    while '  ' in cleaned_text:
        cleaned_text = cleaned_text.replace('  ', ' ')
    
    # Split into sentences/chunks of reasonable length
    words = cleaned_text.split()
    
    samples = []
    chunk_size = len(words) // n_samples
    
    for i in range(n_samples):
        start_idx = i * chunk_size
        end_idx = start_idx + max(chunk_size, min_length)
        chunk = " ".join(words[start_idx:end_idx])
        if len(chunk) >= min_length:
            samples.append(chunk)
    
    return samples


def compute_frequency_distribution(corpus, char_to_idx):
    """
    Compute letter frequency distribution from corpus.
    Returns sorted list of (char_idx, frequency) from most to least frequent.
    """
    N = 27
    freq = np.zeros(N)
    total = 0
    
    for text in corpus:
        for char in text.upper():
            if char in char_to_idx:
                freq[char_to_idx[char]] += 1
                total += 1
    
    freq /= total
    # Return indices sorted by frequency (descending)
    sorted_indices = np.argsort(-freq)
    return sorted_indices, freq


def initialize_emission_frequency_based(encrypted_obs, corpus_freq_order, obs_freq_order, N=27):
    """
    Initialize emission matrix using frequency analysis.
    Maps most frequent observed symbols to most frequent corpus letters.
    """
    B = np.ones((N, N)) * 0.001  # Small baseline probability
    
    # Create initial mapping: most frequent cipher -> most frequent plain
    for i in range(N):
        plain_idx = corpus_freq_order[i]
        cipher_idx = obs_freq_order[i] if i < len(obs_freq_order) else i
        B[plain_idx, cipher_idx] = 0.9  # High probability for frequency match
    
    # Normalize rows
    B /= B.sum(axis=1, keepdims=True)
    return B


def get_observation_frequency(obs, N=27):
    """Get frequency distribution of observations."""
    freq = np.zeros(N)
    for o in obs:
        freq[o] += 1
    freq /= len(obs)
    sorted_indices = np.argsort(-freq)
    return sorted_indices, freq


def save_model(pi, A, B, filepath=MODEL_PATH):
    """Save HMM parameters to file."""
    np.savez(filepath, pi=pi, A=A, B=B)
    print(f"Model saved to {filepath}")


def load_model(filepath=MODEL_PATH):
    """Load HMM parameters from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    data = np.load(filepath)
    pi = data['pi']
    A = data['A']
    B = data['B']
    print(f"Model loaded from {filepath}")
    return pi, A, B


def create_alphabet_mappings():
    """Create mappings between letters and indices (including space)."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '  # 27 symbols including space

    # Mapping dictionaries
    char_to_idx = {char: idx for idx, char in enumerate(alphabet)}
    idx_to_char = {idx: char for idx, char in enumerate(alphabet)}

    # Add lowercase support
    for char in 'abcdefghijklmnopqrstuvwxyz':
        char_to_idx[char] = char_to_idx[char.upper()]

    return char_to_idx, idx_to_char


def encode_text(text, char_to_idx):
    """
    Convert text string to list of numerical indices.
    Removes non-alphabetic characters, converts to uppercase.
    """
    encoded = []
    for char in text.upper():
        if char in char_to_idx:
            encoded.append(char_to_idx[char])
    return encoded


def generate_substitution_key():
    """Generate random substitution cipher key (27 symbols including space)."""
    alphabet = list(range(27))  # 27 symbols
    shuffled = alphabet.copy()
    random.shuffle(shuffled)

    # key[i] = cipher index for plaintext index i
    key = {plain_idx: cipher_idx for plain_idx, cipher_idx in zip(alphabet, shuffled)}
    return key


def encrypt_text(char_to_idx, encoded_text):
    """Apply substitution cipher to encoded text."""
    key = generate_substitution_key()
    encrypted_text = [key[char] for char in encoded_text]
    return encrypted_text, key


def joint_probability_score(predicted_states, observations, A, B, pi):
    """
    predicted_states: List of state indices [s1, s2, ..., sT]
    observations: List of observation indices [o1, o2, ..., oT]
    Returns: P(O, Q | λ) in log space to avoid underflow
    """
    T = len(predicted_states)
    log_score = np.log(pi[predicted_states[0]]) + np.log(B[predicted_states[0], observations[0]])

    for t in range(1, T):
        log_score += np.log(A[predicted_states[t-1], predicted_states[t]])
        log_score += np.log(B[predicted_states[t], observations[t]])

    return log_score  # Higher (less negative) is better


def train_hmm_parameters(corpus, char_to_idx):
    """
    Calculate initial state distribution (pi) and transition matrix (A)
    from the plaintext corpus.
    """
    N = 27  # 27 symbols including space
    A = np.ones((N, N))  # Laplace smoothing
    pi = np.ones(N)      # Laplace smoothing
    
    for sentence in corpus:
        encoded = encode_text(sentence, char_to_idx)
        if not encoded:
            continue
            
        # Update pi
        pi[encoded[0]] += 1
        
        # Update A
        for i in range(len(encoded) - 1):
            curr_s = encoded[i]
            next_s = encoded[i+1]
            A[curr_s, next_s] += 1
            
    # Normalize
    pi /= pi.sum()
    A /= A.sum(axis=1, keepdims=True)
    
    return pi, A
