import pickle
import numpy as np
from functools import partial
import argparse
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
def kmer(S, k=5):
    """k-mer frequency vector (supports 0-4 for ACGTN)"""
    kernel = [5**p for p in range(k)]
    kmers = np.apply_along_axis(partial(np.convolve, v=kernel, mode='valid'), 1, S)
    vectors = np.zeros((S.shape[0], 5**k), dtype=np.float32)
    for d in range(len(S)):
        bbins = np.bincount(kmers[d])
        vectors[d][:len(bbins)] += bbins
    return vectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSR pickle')
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data, 'rb') as f:
        references, queries, labels = pickle.load(f)

    labels = labels.cpu().numpy().astype(int)
    ref_np = references.cpu().numpy()
    qry_np = queries.cpu().numpy()

    print("\n=== K-mer Baseline for CSR (k=2 to 6) ===\n")

    for k in [2, 3, 4, 5, 6]:
        print(f"Computing k={k} vectors...")
        ref_vec = kmer(ref_np, k)
        qry_vec = kmer(qry_np, k)

        dist_matrix = euclidean_distances(qry_vec, ref_vec)

        print(f"k={k} Results:")
        for topk in [1, 5, 10]:
            topk_idx = np.argpartition(dist_matrix, topk, axis=1)[:, :topk]
            correct = np.any(topk_idx == labels[:, None], axis=1)
            acc = correct.mean()
            print(f"   Top-{topk:2d}: {acc:.3f} ({acc*100:.1f}%)")
        print("-" * 50)