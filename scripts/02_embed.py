"""
Step 4: Generate embeddings for each extracted paper.
Uses sentence-transformers (runs locally, no API needed).
Saves all embeddings + metadata to a single JSON file.
"""

import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Paths ──
PROJECT_DIR = Path(__file__).parent.parent
EXTRACT_DIR = PROJECT_DIR / "pdfs" / "extracted"
OUTPUT_FILE = PROJECT_DIR / "output" / "embeddings.json"

# Create output folder if needed
OUTPUT_FILE.parent.mkdir(exist_ok=True)


def load_text(txt_path, max_chars=8000):
    """
    Loads a text file and truncates to max_chars.

    Why truncate? The embedding model has a token limit.
    8000 characters (~2000 words) captures the abstract,
    intro, and key findings — enough for good clustering.
    """
    text = txt_path.read_text(encoding="utf-8")
    return text[:max_chars]
#return to this for more informed details 

def main():
    # Find all extracted text files
    txt_files = sorted(EXTRACT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} extracted papers\n")

    if len(txt_files) == 0:
        print("No text files found! Run 01_extract.py first.")
        return

    # Load the embedding model
    # This downloads ~90MB the first time, then it's cached
    print("Loading embedding model (first time downloads ~90MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    # Process each paper
    papers = []

    for txt_path in tqdm(txt_files, desc="Embedding"):
        text = load_text(txt_path)

        # Generate the embedding — this is the key line
        # It turns text into a list of 384 numbers
        embedding = model.encode(text).tolist()

        papers.append({
            "filename": txt_path.stem,
            "char_count": len(text),
            "embedding": embedding
        })

    # Save everything to one JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(papers, f)

    print(f"\n{'='*40}")
    print(f"Done! {len(papers)} papers embedded.")
    print(f"Each embedding has {len(papers[0]['embedding'])} dimensions.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()