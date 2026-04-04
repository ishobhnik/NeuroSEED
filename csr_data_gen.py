import os
import random
import pickle
import time
import numpy as np
import torch
import multiprocessing as mp
from Bio import Entrez, SeqIO
from tqdm import tqdm
import Levenshtein

Entrez.email = "your@email.com"

REF_TERM = '33175[BioProject] AND "culture collection"[All Fields]'
QRY_TERM = '33175[BioProject] NOT "culture collection"[All Fields]'

N_REFERENCES = 10000
N_QUERIES = 1000
FIXED_LENGTH = 1500
MIN_LEN = 1200
RANDOM_SEED = 42
OUTPUT_PICKLE = "csr_large.pkl"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

NUC_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

def fetch_fasta(term, max_seqs=20000):
    print(f"\nSearching NCBI: {term}")
    handle = Entrez.esearch(db="nuccore", term=term, retmax=max_seqs)
    record = Entrez.read(handle)
    handle.close()
    ids = record["IdList"]
    print(f"Found {len(ids)} sequences. Downloading...")

    sequences = []
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        handle = Entrez.efetch(db="nuccore", id=batch, rettype="fasta", retmode="text")
        for rec in SeqIO.parse(handle, "fasta"):
            seq = str(rec.seq).upper().replace("-", "").replace(" ", "")
            sequences.append(seq)
        handle.close()
        print(f"  Fetched {min(i + batch_size, len(ids))}/{len(ids)}")
        time.sleep(0.34) 
    return sequences


def process_seq(seq, max_n=5):
    """Clean sequence — filter short, high-N, trim/pad to FIXED_LENGTH."""
    if len(seq) < MIN_LEN:
        return None
    seq = ''.join(c if c in 'ACGTU' else 'N' for c in seq)
    if seq.count('N') > max_n:
        return None
    seq = seq[:FIXED_LENGTH]
    seq = seq + 'N' * (FIXED_LENGTH - len(seq))
    return seq


def encode_and_pad(sequences):
    """Convert string sequences to integer tensor (N, FIXED_LENGTH)."""
    encoded = np.full((len(sequences), FIXED_LENGTH), 4, dtype=np.int64)
    for i, seq in enumerate(sequences):
        for j, char in enumerate(seq):
            encoded[i, j] = NUC_TO_INT.get(char, 4)
    return torch.from_numpy(encoded).long()


def compute_distances_row(args):
    """Compute Levenshtein distances from one query to all references."""
    q, refs = args
    return np.array([Levenshtein.distance(q, r) for r in refs], dtype=np.float32)

print("=== Downloading reference sequences (culturable) ===")
raw_refs = fetch_fasta(REF_TERM)
print("=== Downloading query sequences ===")
raw_queries = fetch_fasta(QRY_TERM)

print("\nProcessing sequences...")
processed_refs = [s for seq in raw_refs if (s := process_seq(seq)) is not None]
processed_queries = [s for seq in raw_queries if (s := process_seq(seq)) is not None]
print(f"After filtering: {len(processed_refs):,} refs, {len(processed_queries):,} queries")

references_str = random.sample(processed_refs, min(N_REFERENCES, len(processed_refs)))
print(f"Sampled {len(references_str)} references")

ref_set = set(references_str)
queries_pool = []
for q in random.sample(processed_queries, len(processed_queries)):
    if q not in ref_set:
        queries_pool.append(q)
queries_pool = queries_pool[:N_QUERIES * 2]
print(f"Query pool after dedup: {len(queries_pool)}")

print(f"\nComputing edit distances ({len(queries_pool)} queries x {len(references_str)} refs)...")
args_list = [(q, references_str) for q in queries_pool]

with mp.Pool(processes=mp.cpu_count()) as pool:
    all_distances = list(tqdm(
        pool.imap(compute_distances_row, args_list),
        total=len(queries_pool),
        desc="Computing distances"
    ))

print("\nFiltering ties...")
valid_queries_str = []
valid_labels = []

for q_str, dists in zip(queries_pool, all_distances):
    min_dist = dists.min()
    count_min = (dists == min_dist).sum()
    if count_min == 1:
        valid_queries_str.append(q_str)
        valid_labels.append(int(dists.argmin()))
    if len(valid_queries_str) == N_QUERIES:
        break

print(f"Kept {len(valid_queries_str)} queries after tie filtering")

print("\nEncoding sequences...")
references_tensor = encode_and_pad(references_str)                          
queries_tensor = encode_and_pad(valid_queries_str)                          
labels_tensor = torch.tensor(valid_labels, dtype=torch.float)             

print(f"\nSaving to {OUTPUT_PICKLE}...")
with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump((references_tensor, queries_tensor, labels_tensor), f)

print(f"\n✅ Dataset saved!")
print(f"   References : {references_tensor.shape}")
print(f"   Queries    : {queries_tensor.shape}")
print(f"   Labels     : {labels_tensor.shape}")
print(f"   Format     : integer encoded (N, {FIXED_LENGTH}), long tensor")