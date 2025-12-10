# -*- coding: utf-8 -*-
"""Pre-processing module for HMM cipher breaking.

This module re-exports functions from utils.py for backward compatibility.
The main preprocessing functions (corpus generation, alphabet mappings, 
encoding, encryption) are defined in utils.py.
"""
import numpy as np

# Re-export preprocessing functions from utils
from utils import (
    generate_training_corpus,
    compute_frequency_distribution,
    initialize_emission_frequency_based,
    get_observation_frequency,
    create_alphabet_mappings,
    encode_text,
    generate_substitution_key,
    encrypt_text,
    train_hmm_parameters,
)

__all__ = [
    'generate_training_corpus',
    'compute_frequency_distribution',
    'initialize_emission_frequency_based',
    'get_observation_frequency',
    'create_alphabet_mappings',
    'encode_text',
    'generate_substitution_key',
    'encrypt_text',
    'train_hmm_parameters',
]