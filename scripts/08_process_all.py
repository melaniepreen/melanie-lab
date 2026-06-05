"""
Master pipeline: processes ALL paper folders through the full pipeline.
Extract → Embed → Cluster → Visualize → Notes → Links → Analysis

Run this after adding new papers to any folder.

1. Drop PDFs into the right folder
      (pdfs/ayurveda/, pdfs/tcm/, etc.)

   2. Run the master pipeline:
      python3 scripts/08_process_all.py

   3. Rebuild the visualization + notes:
      python3 scripts/04_visualize.py
      python3 scripts/05_obsidian_notes.py
      python3 scripts/06_link_notes.py
      python3 scripts/07_cross_analysis.py

   4. Open the map + Obsidian and explore
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

load_dotenv()
PROJECT_DIR = Path(__file__).parent.parent
PDF_BASE = PROJECT_DIR / "pdfs"
EXTRACT_DIR = PDF_BASE / "extracted"
OUTPUT_DIR = PROJECT_DIR / "output"
NOTES_DIR = PROJECT_DIR / "notes_vault" / "papers"

EXTRACT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

OBSIDIAN_VAULT = "notes_vault"

# All paper folders to process
PAPER_FOLDERS = [
    "papers-meditation",
    "papers-sound",
]


def step_1_extract():
    """Extract text from all PDFs across all folders."""
    from unstructured.partition.pdf import partition_pdf

    print("\n" + "=" * 50)
    print("STEP 1: EXTRACTING TEXT FROM PDFs")
    print("=" * 50)

    all_pdfs = []
    for folder in PAPER_FOLDERS:
        folder_path = PDF_BASE / folder
        if folder_path.exists():
            pdfs = sorted(folder_path.glob("*.pdf"))
            all_pdfs.extend(pdfs)
            print(f"  {folder}: {len(pdfs)} PDFs")

    print(f"\n  Total: {len(all_pdfs)} PDFs")

    success = 0
    for pdf_path in tqdm(all_pdfs, desc="Extracting"):
        # Add tradition prefix to avoid filename collisions
        tradition = pdf_path.parent.name
        safe_name = f"{tradition}__{pdf_path.stem}"
        txt_path = EXTRACT_DIR / (safe_name + ".txt")

        if txt_path.exists():
            success += 1
            continue

        try:
            elements = partition_pdf(str(pdf_path))
            text = "\n\n".join([str(el) for el in elements])
            if len(text.strip()) > 100:
                txt_path.write_text(text, encoding="utf-8")
                success += 1
        except Exception as e:
            print(f"\n  Error: {pdf_path.name}: {e}")

    print(f"  Extracted: {success}")
    return success


def step_2_embed():
    """Generate embeddings for all extracted papers."""
    print("\n" + "=" * 50)
    print("STEP 2: GENERATING EMBEDDINGS")
    print("=" * 50)

    txt_files = sorted(EXTRACT_DIR.glob("*.txt"))
    print(f"  Found {len(txt_files)} text files")

    # Check for existing embeddings
    emb_file = OUTPUT_DIR / "embeddings.json"
    existing = {}
    if emb_file.exists():
        with open(emb_file) as f:
            for p in json.load(f):
                existing[p["filename"]] = p

    # Find new papers that need embedding
    new_files = [f for f in txt_files if f.stem not in existing]
    print(f"  Already embedded: {len(existing)}")
    print(f"  New to embed: {len(new_files)}")

    if not new_files:
        print("  Nothing new to embed.")
        return len(existing)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    for txt_path in tqdm(new_files, desc="Embedding"):
        text = txt_path.read_text(encoding="utf-8")[:8000]
        embedding = model.encode(text).tolist()

        # Extract tradition from filename prefix
        parts = txt_path.stem.split("__", 1)
        tradition = parts[0] if len(parts) > 1 else "unknown"

        existing[txt_path.stem] = {
            "filename": txt_path.stem,
            "tradition": tradition,
            "char_count": len(text),
            "embedding": embedding,
        }

    # Save all embeddings
    papers = list(existing.values())
    with open(emb_file, "w") as f:
        json.dump(papers, f)

    print(f"  Total embedded: {len(papers)}")
    return len(papers)


def step_3_cluster():
    """UMAP + HDBSCAN on all embeddings."""
    print("\n" + "=" * 50)
    print("STEP 3: CLUSTERING")
    print("=" * 50)

    with open(OUTPUT_DIR / "embeddings.json") as f:
        papers = json.load(f)

    filenames = [p["filename"] for p in papers]
    traditions = [p.get("tradition", "unknown") for p in papers]
    embeddings = np.array([p["embedding"] for p in papers])

    print(f"  Papers: {len(filenames)}")
    print(f"  Traditions: {set(traditions)}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=12,
        min_dist=0.2,
        spread=1.5,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=4,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Clusters found: {n_clusters}")

    df = pd.DataFrame({
        "filename": filenames,
        "tradition": traditions,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels,
    })

    df.to_csv(OUTPUT_DIR / "corpus_map.csv", index=False)
    print(f"  Saved corpus_map.csv")

    # Print tradition distribution per cluster
    print(f"\n  Cluster composition:")
    for c in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == c]
        trad_counts = subset["tradition"].value_counts().to_dict()
        label = "unclustered" if c == -1 else f"cluster {c}"
        print(f"    {label}: {dict(trad_counts)}")

    return df


def main():
    print("=" * 50)
    print("MELANIE LAB — FULL PIPELINE")
    print("=" * 50)

    # Count available papers
    total = 0
    for folder in PAPER_FOLDERS:
        folder_path = PDF_BASE / folder
        if folder_path.exists():
            count = len(list(folder_path.glob("*.pdf")))
            total += count

    print(f"\nTotal PDFs across all folders: {total}")

    if total == 0:
        print("No PDFs found! Add papers to the /pdfs/ subfolders.")
        return

    step_1_extract()
    step_2_embed()
    df = step_3_cluster()

    print(f"\n{'=' * 50}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 50}")
    print(f"\nNext steps:")
    print(f"  1. Run: python3 scripts/04_visualize.py  (rebuild the map)")
    print(f"  2. Run: python3 scripts/05_obsidian_notes.py  (generate notes)")
    print(f"  3. Run: python3 scripts/06_link_notes.py  (add wiki-links)")
    print(f"  4. Run: python3 scripts/07_cross_analysis.py  (cluster analysis)")
    print(f"\nOr run them all in sequence:")
    print(f"  python3 scripts/04_visualize.py && python3 scripts/05_obsidian_notes.py && python3 scripts/06_link_notes.py && python3 scripts/07_cross_analysis.py")


if __name__ == "__main__":
    main()