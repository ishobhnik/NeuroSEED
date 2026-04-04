import os
import random
import pickle
import numpy as np
import torch
import multiprocessing as mp
from Bio import SeqIO
import argparse
from tqdm import tqdm
import Levenshtein   

NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
FIXED_LENGTH = 1500      
MIN_LEN = 1200

def parse_and_clean_fasta(fasta_path, max_n_allowed=5):
    print("Parsing and cleaning FASTA...")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    
    unique_seqs = set()
    clean_seqs = []
    
    for r in records:
        seq = str(r.seq).upper()
        if (seq not in unique_seqs and 
            seq.count('N') <= max_n_allowed and 
            len(seq) >= MIN_LEN):
            seq = seq[:FIXED_LENGTH]
            unique_seqs.add(seq)
            clean_seqs.append(seq)
    
    print(f"Kept {len(clean_seqs)} unique clean sequences.")
    print(f"All sequences truncated/padded to {FIXED_LENGTH} bp.")
    return clean_seqs

def encode_and_pad(sequences, fixed_length):
    encoded = np.full((len(sequences), fixed_length), 4, dtype=np.int64)
    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq):
            encoded[i, j] = NUC_TO_INT.get(char, 4)
    return torch.from_numpy(encoded).long()

def compute_pairwise_row(args):
    """Pure Levenshtein edit distance (same as synthetic dataset)"""
    i, seq_i, all_seqs = args
    row_distances = np.zeros(len(all_seqs), dtype=np.float32)
    
    for j in range(i, len(all_seqs)):
        if i == j:
            row_distances[j] = 0.0
        else:
            row_distances[j] = Levenshtein.distance(seq_i, all_seqs[j])
    
    return i, row_distances

def build_split(name, sequences, fixed_length, num_cores):
    print(f"--- Building {name} Split ({len(sequences)} sequences) ---")
    seq_tensor = encode_and_pad(sequences, fixed_length)
    
    dist_matrix = np.zeros((len(sequences), len(sequences)), dtype=np.float32)
    mp_args = [(i, seq, sequences) for i, seq in enumerate(sequences)]
    
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(compute_pairwise_row, mp_args),
                            total=len(mp_args),
                            desc=f"Calculating {name} Levenshtein Distances",
                            unit="row"))
    
    for i, row in results:
        dist_matrix[i, :] = row
    
    i_lower = np.tril_indices(len(sequences), -1)
    dist_matrix[i_lower] = dist_matrix.T[i_lower]
    
    dist_tensor = torch.from_numpy(dist_matrix).float()
    return seq_tensor, dist_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Greengenes 1600 bp - Pure Levenshtein")
    parser.add_argument('--fasta', type=str, required=True, help='Path to Greengenes .fasta')
    parser.add_argument('--out', type=str, default='./data/edit_greengenes_1600_levenshtein.pkl')
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--val_size', type=int, default=500)
    parser.add_argument('--test_size', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cores', type=int, default=mp.cpu_count())
    args = parser.parse_args()

    random.seed(args.seed)

    all_seqs = parse_and_clean_fasta(args.fasta)
    
    total_needed = args.train_size + args.val_size + args.test_size
    if len(all_seqs) < total_needed:
        raise ValueError(f"Need {total_needed} sequences, only found {len(all_seqs)}")

    random.shuffle(all_seqs)
    train_seqs = all_seqs[:args.train_size]
    val_seqs   = all_seqs[args.train_size : args.train_size + args.val_size]
    test_seqs  = all_seqs[args.train_size + args.val_size : total_needed]

    train_seq_t, train_dist_t = build_split('Train', train_seqs, FIXED_LENGTH, args.cores)
    val_seq_t,   val_dist_t   = build_split('Validation', val_seqs, FIXED_LENGTH, args.cores)
    test_seq_t,  test_dist_t  = build_split('Test', test_seqs, FIXED_LENGTH, args.cores)

    sequences_dict = {'train': train_seq_t, 'val': val_seq_t, 'test': test_seq_t}
    distances_dict = {'train': train_dist_t, 'val': val_dist_t, 'test': test_dist_t}
    final_tuple = (sequences_dict, distances_dict)

    print(f"Saving dataset to {args.out}...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(final_tuple, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Done! Pure Levenshtein dataset ready.")