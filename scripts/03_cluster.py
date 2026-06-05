"""
Step 5: Reduce embeddings to 2D with UMAP, cluster with HDBSCAN.
Reads embeddings.json, outputs corpus_map.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import umap
import hdbscan

# ── Paths ──
PROJECT_DIR = Path(__file__).parent.parent
INPUT_FILE = PROJECT_DIR / "output" / "embeddings.json"
OUTPUT_FILE = PROJECT_DIR / "output" / "corpus_map.csv"

def main():
    # Load embeddings
    print("Loading embeddings...")
    with open(INPUT_FILE) as f:
        papers = json.load(f)

    filenames = [p["filename"] for p in papers]
    embeddings = np.array([p["embedding"] for p in papers])

    print(f"Loaded {len(filenames)} papers, {embeddings.shape[1]} dimensions each.\n")

    # ── UMAP: 384 dimensions → 2 dimensions ──
    # Think of this as squashing a 384-dimensional cloud of points
    # onto a flat 2D surface, while preserving which points are
    # close to each other. Similar papers stay near each other.
    print("Running UMAP (this takes 10-30 seconds)...")
    reducer = umap.UMAP(
        n_components=2,       # output: 2D (x, y)
        n_neighbors=15,       # how many neighbors to consider
        min_dist=0.1,         # how tightly points can cluster
        metric="cosine",      # best for text embeddings
        random_state=42       # reproducible results
    )
    coords = reducer.fit_transform(embeddings)
    print("UMAP done.\n")

    # ── HDBSCAN: find clusters ──
    # This automatically finds groups of similar papers.
    # You don't tell it how many clusters — it figures it out.
    # Papers that don't fit any cluster get labeled -1 (noise).
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,       # minimum papers to form a cluster
        min_samples=2,            # how conservative (lower = more clusters)
        metric="euclidean"        # distance on the 2D UMAP coords
    )
    labels = clusterer.fit_predict(coords)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"Found {n_clusters} clusters, {n_noise} unclustered papers.\n")

    # ── Build the output table ──
    df = pd.DataFrame({
        "filename": filenames,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels
    })

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"{'='*40}")
    print(f"Done! Saved to: {OUTPUT_FILE}")
    print(f"\nPreview:")
    print(df.head(10).to_string(index=False))

    # Show cluster distribution
    print(f"\nCluster sizes:")
    for c in sorted(df["cluster"].unique()):
        count = len(df[df["cluster"] == c])
        label = "unclustered" if c == -1 else f"cluster {c}"
        print(f"  {label}: {count} papers")


if __name__ == "__main__":
    main()