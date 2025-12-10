# -*- coding: utf-8 -*-
"""Simple Tkinter GUI for HMM Substitution Cipher Breaker."""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
from datetime import datetime
import numpy as np

# Import HMM modules
from utils import *
from Viterbi import *
from BaumWelch import *
from PreProcessing import *


class HMMCipherBreakerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HMM Substitution Cipher Breaker")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Variables
        self.input_text = ""
        self.is_running = False
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="HMM Substitution Cipher Breaker", 
                                font=("Helvetica", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Top frame for file upload and controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File upload button
        self.upload_btn = ttk.Button(top_frame, text="📂 Upload Text File", 
                                      command=self.upload_file)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # File label
        self.file_label = ttk.Label(top_frame, text="No file selected", 
                                     foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Run button
        self.run_btn = ttk.Button(top_frame, text="▶ Run Decryption", 
                                   command=self.run_decryption, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        self.clear_btn = ttk.Button(top_frame, text="🗑 Clear", 
                                     command=self.clear_output)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Training samples
        ttk.Label(params_frame, text="Training Samples:").pack(side=tk.LEFT, padx=(0, 5))
        self.samples_var = tk.StringVar(value="1000")
        samples_entry = ttk.Entry(params_frame, textvariable=self.samples_var, width=8)
        samples_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        # Restarts
        ttk.Label(params_frame, text="Restarts:").pack(side=tk.LEFT, padx=(0, 5))
        self.restarts_var = tk.StringVar(value="3")
        restarts_entry = ttk.Entry(params_frame, textvariable=self.restarts_var, width=5)
        restarts_entry.pack(side=tk.LEFT, padx=(0, 15))
        
        # Iterations
        ttk.Label(params_frame, text="Iterations:").pack(side=tk.LEFT, padx=(0, 5))
        self.iterations_var = tk.StringVar(value="30")
        iterations_entry = ttk.Entry(params_frame, textvariable=self.iterations_var, width=5)
        iterations_entry.pack(side=tk.LEFT)
        
        # Output text area
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                      font=("Consolas", 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X)
        
    def upload_file(self):
        """Open file dialog and load text file."""
        file_path = filedialog.askopenfilename(
            title="Select Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.input_text = f.read()
                
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Loaded: {filename}", foreground="green")
                self.run_btn.config(state=tk.NORMAL)
                self.log(f"✅ Loaded file: {filename}")
                self.log(f"   Characters: {len(self.input_text)}")
                self.status_var.set(f"File loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                
    def log(self, message):
        """Add message to output text area."""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
        
    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.status_var.set("Ready")
        
    def run_decryption(self):
        """Run the HMM decryption in a separate thread."""
        if self.is_running:
            return
            
        if not self.input_text.strip():
            messagebox.showwarning("Warning", "Please upload a text file first.")
            return
            
        # Get parameters
        try:
            n_samples = int(self.samples_var.get())
            n_restarts = int(self.restarts_var.get())
            n_iterations = int(self.iterations_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values. Please enter integers.")
            return
            
        # Disable controls
        self.is_running = True
        self.run_btn.config(state=tk.DISABLED)
        self.upload_btn.config(state=tk.DISABLED)
        self.progress.start()
        
        # Run in separate thread
        thread = threading.Thread(target=self.decrypt_thread, 
                                  args=(n_samples, n_restarts, n_iterations))
        thread.daemon = True
        thread.start()
        
    def decrypt_thread(self, n_samples, n_restarts, n_iterations):
        """Thread function for decryption."""
        try:
            self.log("\n" + "=" * 50)
            self.log("HMM Substitution Cipher Breaker")
            self.log("=" * 50)
            
            # 1. Setup
            self.status_var.set("Setting up alphabet mappings...")
            self.log("\n[Setup] Creating alphabet mappings...")
            char_to_idx, idx_to_char = create_alphabet_mappings()
            states = list(range(27))
            N = 27
            
            # 2. Train reference model
            self.status_var.set("Training reference model from corpus...")
            self.log("\n[Reference Model] Training from corpus...")
            self.log(f"   Generating {n_samples} training samples...")
            
            plaintext_corpus = generate_training_corpus(n_samples=n_samples, min_length=50)
            self.log(f"   ✅ Generated {len(plaintext_corpus)} training samples")
            
            self.log("   Computing transition matrix and start probabilities...")
            pi, A = train_hmm_parameters(plaintext_corpus, char_to_idx)
            self.log("   ✅ Reference Transition Matrix and Start Probabilities computed.")
            
            corpus_freq_order, corpus_freq = compute_frequency_distribution(plaintext_corpus, char_to_idx)
            self.log("   ✅ Corpus frequency distribution computed.")
            self.log(f"   Top 5 most frequent letters: {[idx_to_char[i] for i in corpus_freq_order[:5]]}")
            
            # 3. Encryption
            self.status_var.set("Encrypting message...")
            self.log("\n[Test Data] Encrypting message...")
            
            secret_message = self.input_text
            encoded_msg = encode_text(secret_message, char_to_idx)
            
            if len(encoded_msg) == 0:
                raise ValueError("Input text contains no valid characters (A-Z or space)")
                
            encrypted_msg, true_key = encrypt_text(char_to_idx, encoded_msg)
            encrypted_text_chars = ''.join([idx_to_char[c] for c in encrypted_msg])
            
            self.log(f"   Message Length: {len(encoded_msg)} characters")
            self.log(f"   Plain Text: {secret_message.upper()[:100]}...")
            self.log(f"   Encrypted:  {encrypted_text_chars[:100]}...")
            
            obs_freq_order, obs_freq = get_observation_frequency(encrypted_msg, N=27)
            
            # 4. Training with restarts
            self.status_var.set("Running Baum-Welch algorithm...")
            self.log(f"\n[Training] Running Baum-Welch with {n_restarts} restarts...")
            
            best_B = None
            best_log_likelihood = float('-inf')
            
            for restart in range(n_restarts):
                self.status_var.set(f"Restart {restart + 1}/{n_restarts}...")
                self.log(f"\n--- Restart {restart + 1}/{n_restarts} ---")
                
                if restart == 0:
                    self.log("   Using frequency-based initialization")
                    B_init = initialize_emission_frequency_based(
                        encrypted_msg, corpus_freq_order, obs_freq_order, N
                    )
                else:
                    self.log("   Using random initialization with frequency bias")
                    B_init = np.random.rand(N, N)
                    for i in range(N):
                        plain_idx = corpus_freq_order[i]
                        cipher_idx = obs_freq_order[i] if i < len(obs_freq_order) else i
                        B_init[plain_idx, cipher_idx] += np.random.rand() * 0.5
                    B_init /= B_init.sum(axis=1, keepdims=True)
                
                learned_B_candidate = Baum_Welch(
                    encrypted_msg, states, pi, A, B_init, n_iter=n_iterations
                )
                
                alpha = forward(encrypted_msg, states, pi, A, learned_B_candidate)
                final_alpha = alpha[-1]
                max_val = np.max(final_alpha)
                log_likelihood = max_val + np.log(np.sum(np.exp(final_alpha - max_val)))
                
                self.log(f"   Final Log Likelihood: {log_likelihood:.4f}")
                
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_B = learned_B_candidate
                    self.log(f"   ✅ New best model found!")
                    
            self.log(f"\n[Best Model] Log Likelihood = {best_log_likelihood:.4f}")
            
            # 5. Decode with Viterbi
            self.status_var.set("Decoding with Viterbi algorithm...")
            self.log("\n[Decoding] Using Viterbi algorithm...")
            
            decoded_indices = Viterbi_decoding(encrypted_msg, states, pi, A, best_B, verbose=False)
            decoded_text = "".join([idx_to_char[idx] for idx in decoded_indices])
            
            # 6. Calculate accuracy
            matches = sum(1 for e, d in zip(encoded_msg, decoded_indices) if e == d)
            accuracy = matches / len(encoded_msg) * 100 if len(encoded_msg) > 0 else 0
            
            self.log(f"   ✅ Decoding complete")
            self.log(f"   Accuracy: {accuracy:.2f}% ({matches}/{len(encoded_msg)} correct)")
            
            # 7. Display results
            self.log("\n" + "=" * 50)
            self.log("RESULTS")
            self.log("=" * 50)
            self.log(f"\nPlain Text:\n{secret_message.upper()}")
            self.log(f"\nEncrypted Text:\n{encrypted_text_chars}")
            self.log(f"\nDecrypted Text:\n{decoded_text}")
            self.log(f"\nAccuracy: {accuracy:.2f}%")
            self.log(f"Best Log Likelihood: {best_log_likelihood:.4f}")
            
            # 8. Save to Test folder
            self.status_var.set("Saving results...")
            test_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Test")
            os.makedirs(test_folder, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"result_{timestamp}.txt"
            output_path = os.path.join(test_folder, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("HMM Substitution Cipher Breaker - Results\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("-" * 50 + "\n")
                f.write("PARAMETERS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Training Samples: {n_samples}\n")
                f.write(f"Restarts: {n_restarts}\n")
                f.write(f"Iterations per Restart: {n_iterations}\n\n")
                f.write("-" * 50 + "\n")
                f.write("PLAINTEXT (Original Message):\n")
                f.write("-" * 50 + "\n")
                f.write(f"{secret_message.upper()}\n\n")
                f.write("-" * 50 + "\n")
                f.write("ENCRYPTED TEXT (After Substitution Cipher):\n")
                f.write("-" * 50 + "\n")
                f.write(f"{encrypted_text_chars}\n\n")
                f.write("-" * 50 + "\n")
                f.write("DECRYPTED TEXT (HMM Output):\n")
                f.write("-" * 50 + "\n")
                f.write(f"{decoded_text}\n\n")
                f.write("=" * 50 + "\n")
                f.write("STATISTICS:\n")
                f.write("=" * 50 + "\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n")
                f.write(f"Correct Characters: {matches}/{len(encoded_msg)}\n")
                f.write(f"Best Log Likelihood: {best_log_likelihood:.4f}\n")
                
            self.log(f"\n📁 Results saved to: {output_path}")
            
            self.status_var.set(f"✅ Complete! Accuracy: {accuracy:.2f}%")
            messagebox.showinfo("Success", 
                              f"Decryption complete!\n\nAccuracy: {accuracy:.2f}%\n\nResults saved to:\n{output_path}")
            
        except Exception as e:
            self.log(f"\n❌ Error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
            
        finally:
            # Re-enable controls
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.upload_btn.config(state=tk.NORMAL))
            self.root.after(0, self.progress.stop)


def main():
    root = tk.Tk()
    app = HMMCipherBreakerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
