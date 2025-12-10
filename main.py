import numpy as np
import random
import nltk
import os
from datetime import datetime
from utils import *
from Viterbi import *
from BaumWelch import *
from PreProcessing import *

def main_encrypt_decrypt(input_text,
                          output_file='results.txt',
                          n_training_samples=1000,
                          n_restarts=3,
                          n_iterations=30,
                          verbose=True):
    """
    Main function to encrypt text using substitution cipher and decrypt using HMM
    with multiple random restarts to find the best model.

    Parameters:
    -----------
    input_text : str
        The plaintext to encrypt and decrypt
    output_file : str
        File to save results
    n_training_samples : int
        Number of training samples to generate
    n_restarts : int
        Number of random restarts for Baum-Welch training
    n_iterations : int
        Number of Baum-Welch iterations per restart
    verbose : bool
        Print detailed progress

    Returns:
    --------
    dict with keys:
        - 'original_text': input text
        - 'encrypted_text': encrypted text (as characters)
        - 'encrypted_indices': encrypted numerical indices
        - 'decrypted_text': decoded text
        - 'encryption_key': substitution cipher key used
        - 'accuracy': character-level accuracy of decryption
        - 'best_log_likelihood': best log likelihood achieved
    """

    if verbose:
        print("=" * 50)
        print("HMM Substitution Cipher Breaker")
        print("=" * 50)

    # ========================
    # 1. SETUP DATA AND MAPPINGS
    # ========================
    if verbose:
        print("\n[Setup] Creating alphabet mappings...")

    char_to_idx, idx_to_char = create_alphabet_mappings()
    states = list(range(27))
    N = 27

    # ========================
    # 2. TRAIN REFERENCE MODEL (A, pi) FROM CORPUS
    # ========================
    if verbose:
        print("\n[Reference Model] Training from corpus...")
        print(f"   Generating {n_training_samples} training samples...")

    plaintext_corpus = generate_training_corpus(n_samples=n_training_samples, min_length=50)

    if verbose:
        print(f"   ✅ Generated {len(plaintext_corpus)} training samples")
        print("   Computing transition matrix and start probabilities...")

    pi, A = train_hmm_parameters(plaintext_corpus, char_to_idx)

    if verbose:
        print("   ✅ Reference Transition Matrix and Start Probabilities computed.")

    # Compute corpus frequency for initialization
    corpus_freq_order, corpus_freq = compute_frequency_distribution(plaintext_corpus, char_to_idx)

    if verbose:
        print("   ✅ Corpus frequency distribution computed.")
        print(f"   Top 5 most frequent letters: {[idx_to_char[i] for i in corpus_freq_order[:5]]}")

    # ========================
    # 3. PREPARE TEST DATA (ENCRYPTION)
    # ========================
    if verbose:
        print(f"\n[Test Data] Encrypting message...")

    secret_message = input_text
    encoded_msg = encode_text(secret_message, char_to_idx)

    if len(encoded_msg) == 0:
        raise ValueError("Input text contains no valid characters (A-Z or space)")

    encrypted_msg, true_key = encrypt_text(char_to_idx, encoded_msg)
    encrypted_text_chars = ''.join([idx_to_char[c] for c in encrypted_msg])

    if verbose:
        print(f"   Message Length: {len(encoded_msg)} characters")
        print(f"   Plain Text:            {secret_message.upper()}")
        print(f"   Text After Encryption: {encrypted_text_chars}")

    # Compute observation frequency
    obs_freq_order, obs_freq = get_observation_frequency(encrypted_msg, N=27)

    if verbose:
        print(f"   Top 5 most frequent encrypted symbols: {obs_freq_order[:5]}")

    # ========================
    # 4. TRAIN WITH MULTIPLE RESTARTS
    # ========================
    if verbose:
        print(f"\n[Training] Running Baum-Welch with frequency-based + random restarts...")
        print(f"   Restarts: {n_restarts}, Iterations per restart: {n_iterations}")

    best_B = None
    best_log_likelihood = float('-inf')

    for restart in range(n_restarts):
        if verbose:
            print(f"\n--- Restart {restart + 1}/{n_restarts} ---")

        # Initialize emission matrix B
        if restart == 0:
            # First restart: use frequency-based initialization
            if verbose:
                print("   Using frequency-based initialization")
            B_init = initialize_emission_frequency_based(
                encrypted_msg, corpus_freq_order, obs_freq_order, N
            )
        else:
            # Subsequent restarts: random initialization with frequency bias
            if verbose:
                print("   Using random initialization with frequency bias")
            B_init = np.random.rand(N, N)
            for i in range(N):
                plain_idx = corpus_freq_order[i]
                cipher_idx = obs_freq_order[i] if i < len(obs_freq_order) else i
                B_init[plain_idx, cipher_idx] += np.random.rand() * 0.5
            B_init /= B_init.sum(axis=1, keepdims=True)

        # Run Baum-Welch (updates only B, keeps A and pi fixed)
        learned_B_candidate = Baum_Welch(
            encrypted_msg, states, pi, A, B_init, n_iter=n_iterations
        )

        # Compute log likelihood for this model
        alpha = forward(encrypted_msg, states, pi, A, learned_B_candidate)
        final_alpha = alpha[-1]
        max_val = np.max(final_alpha)
        log_likelihood = max_val + np.log(np.sum(np.exp(final_alpha - max_val)))

        if verbose:
            print(f"   Final Log Likelihood: {log_likelihood:.4f}")

        # Keep the best model
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_B = learned_B_candidate
            if verbose:
                print(f"   ✅ New best model found!")

    if verbose:
        print(f"\n[Best Model] Log Likelihood = {best_log_likelihood:.4f}")

    # ========================
    # 5. DECODE WITH VITERBI
    # ========================
    if verbose:
        print("\n[Decoding] Using Viterbi algorithm to decode encrypted message...")

    decoded_indices = Viterbi_decoding(encrypted_msg, states, pi, A, best_B, verbose=False)
    decoded_text = "".join([idx_to_char[idx] for idx in decoded_indices])

    # ========================
    # 6. CALCULATE ACCURACY
    # ========================
    matches = sum(1 for e, d in zip(encoded_msg, decoded_indices) if e == d)
    accuracy = matches / len(encoded_msg) * 100 if len(encoded_msg) > 0 else 0

    if verbose:
        print(f"   ✅ Decoding complete")
        print(f"   Accuracy: {accuracy:.2f}% ({matches}/{len(encoded_msg)} correct)")

    # ========================
    # 7. PRINT RESULTS
    # ========================
    if verbose:
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Plain Text:                    {secret_message.upper()}")
        print(f"Text After Encryption:         {encrypted_text_chars}")
        print(f"Final Result After Decryption: {decoded_text}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Best Log Likelihood: {best_log_likelihood:.4f}")
    return {
        'original_text': secret_message.upper(),
        'encrypted_text': encrypted_text_chars,
        'encrypted_indices': encrypted_msg,
        'decrypted_text': decoded_text,
        'encryption_key': true_key,
        'accuracy': accuracy,
        'matches': matches,
        'total_chars': len(encoded_msg),
        'best_log_likelihood': best_log_likelihood
    }


sample_text = """ Detachment is a 2011 American psychological drama film directed by Tony Kaye and written by Carl Lund Its story follows Henry Barthes a highschool substitute teacher who becomes a role model to his students and others It stars Adrien Brody Marcia Gay Harden Christina Hendricks William Petersen Bryan Cranston Tim Blake Nelson Betty Kaye Sami Gayle Lucy Liu Blythe Danner and James Caan

Produced by Greg Shapiro Carl Lund Bingo Gubelmann Austin Stark Benji Kohn and Chris Papavasiliou the film was released on March 16 2012 to mixed reviews

Plot
Substitute teacher Henry Barthes is called in for a onemonth assignment teaching English classes at a high school with many students performing at a low grade level During his first class he observes many acts of hostility and antagonism including two students verbally harassing pupil Meredith with whom he becomes acquainted after class

After school Henry is called in to the care facility to see his grandfather as he has locked himself in the bathroom and will not come out He frequently believes Henry is Patricia Henrys mother On the bus ride home Henry sees teenage runaway Erica having sex for money then getting hit by a man who refuses to pay her Erica attempts to convince Henry to have sex with her which he refuses

The next day Henry reads aloud to the class through the essays from earlier After reading an anonymous essay which is assumed to be Merediths he becomes aware of her struggles with suicidal thoughts On the way home Henry again runs into Erica who he invites up to his apartment and allows her to stay temporarily

Later fellow teacher Sarah asks Henry out Henry returns home late and Erica is still awake waiting for him She is upset that he went out without telling her to which he informs her that she cannot expect him to tell her such things

Henry is later called into the care facility because his grandfather has become gravely ill Henry pretends to be Patricia and tells his grandfather hes done nothing wrong and can allow himself to let go After the visit Henry and Erica go to the park where Henry details Patricias suicide He also implies that his grandfather had sexually abused Patricia but says that he never felt unsafe around either of them

Back at school Meredith shows Henry an artwork that she made for him She opens up about her struggles and becomes visibly upset hugging Henry and asking him to console her Sarah walks in on them causing Meredith to run away and she accuses Henry of touching Meredith inappropriately Henry insists that he was just comforting her but the accusation leads to a surge of memories about his grandfather and Patricia and hes overwhelmed by panic

Later that day Henry is informed of his grandfathers death Feeling overwhelmed after all thats happened Henry tells Erica he can no longer take care of her and has social services take her to a foster home She pleads with Henry to let her stay but he reluctantly maintains his stance

As Henrys assignment comes to an end his students show their appreciation for his work During a break Henry tries to apologise to Meredith for earlier but she quickly dismisses the matter She then intentionally eats a cupcake that she poisoned ending her life Her death leads Henry into a state of reflection and he decides to go visit Erica in the foster care facility for the first time they were separated She euphorically embraces him"""
# ==========================================
# Run the complete pipeline
results = main_encrypt_decrypt(
        input_text=sample_text,
        output_file='results.txt',
        n_training_samples=1000,
        n_restarts=3,
        n_iterations=30,
        verbose=True
    )

print("\n" + "=" * 50)
print("✅ Process complete!")
print("=" * 50)
print(f"Accuracy: {results['accuracy']:.2f}%")

# ==========================================
# Save results to Test folder
# ==========================================
test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test")
os.makedirs(test_folder, exist_ok=True)

# Create filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"result_{timestamp}.txt"
output_path = os.path.join(test_folder, output_filename)

# Write results to file
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\n")
    f.write("HMM Substitution Cipher Breaker - Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("-" * 50 + "\n")
    f.write("PLAINTEXT (Original Message):\n")
    f.write("-" * 50 + "\n")
    f.write(f"{results['original_text']}\n\n")
    f.write("-" * 50 + "\n")
    f.write("ENCRYPTED TEXT (After Substitution Cipher):\n")
    f.write("-" * 50 + "\n")
    f.write(f"{results['encrypted_text']}\n\n")
    f.write("-" * 50 + "\n")
    f.write("DECRYPTED TEXT (HMM Output):\n")
    f.write("-" * 50 + "\n")
    f.write(f"{results['decrypted_text']}\n\n")
    f.write("=" * 50 + "\n")
    f.write("STATISTICS:\n")
    f.write("=" * 50 + "\n")
    f.write(f"Accuracy: {results['accuracy']:.2f}%\n")
    f.write(f"Correct Characters: {results['matches']}/{results['total_chars']}\n")
    f.write(f"Best Log Likelihood: {results['best_log_likelihood']:.4f}\n")

print(f"\n📁 Results saved to: {output_path}")