"""
Step 6.5: Add [[wiki-links]] between related papers in Obsidian notes,
and add cluster tags so you can color the graph view by cluster.

How it works:
1. Reads corpus_map.csv to know which cluster each paper belongs to
2. For each paper, finds its nearest neighbors (from embeddings)
3. Adds [[links]] to those neighbors in the note
4. Adds a cluster tag for Obsidian graph coloring
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ── Config ──
PROJECT_DIR = Path(__file__).parent.parent
NOTES_DIR = PROJECT_DIR / "notes_vault" / "papers"
EMBEDDINGS_FILE = PROJECT_DIR / "output" / "embeddings.json"
CLUSTER_FILE = PROJECT_DIR / "output" / "corpus_map.csv"

# How many links per paper (its closest neighbors)
N_LINKS = 5


def load_data():
    """Load embeddings and cluster assignments."""
    with open(EMBEDDINGS_FILE) as f:
        papers = json.load(f)

    df = pd.read_csv(CLUSTER_FILE)

    filenames = [p["filename"] for p in papers]
    embeddings = np.array([p["embedding"] for p in papers])

    # Build a lookup: filename → cluster
    cluster_lookup = dict(zip(df["filename"], df["cluster"]))

    return filenames, embeddings, cluster_lookup


def find_nearest_neighbors(filenames, embeddings, n_neighbors=N_LINKS):
    """
    For each paper, find its N most similar papers
    using cosine similarity on the embeddings.
    """
    # Compute similarity matrix (every paper vs every paper)
    sim_matrix = cosine_similarity(embeddings)

    neighbors = {}
    for i, filename in enumerate(filenames):
        # Get similarity scores for this paper
        scores = sim_matrix[i]

        # Sort by similarity (highest first), skip self (index 0)
        ranked = np.argsort(scores)[::-1]
        top_neighbors = []
        for j in ranked:
            if j != i and len(top_neighbors) < n_neighbors:
                top_neighbors.append({
                    "filename": filenames[j],
                    "similarity": float(scores[j])
                })
        neighbors[filename] = top_neighbors

    return neighbors


def get_cluster_name(cluster_id):
    """Convert cluster number to a tag-friendly name."""
    if cluster_id == -1:
        return "unclustered"
    return f"cluster_{cluster_id}"


def update_note(note_path, filename, neighbors, cluster_lookup):
    """
    Updates an existing Obsidian note by:
    1. Adding cluster tag to frontmatter
    2. Replacing the Links section with actual [[wiki-links]]
    """
    if not note_path.exists():
        return False

    content = note_path.read_text(encoding="utf-8")
    cluster_id = cluster_lookup.get(filename, -1)
    cluster_tag = get_cluster_name(cluster_id)

    # ── 1. Add cluster tag to frontmatter ──
    # Find the closing --- of frontmatter
    parts = content.split("---", 2)
    if len(parts) >= 3:
        frontmatter = parts[1]

        # Add cluster field if not already there
        if "cluster:" not in frontmatter:
            frontmatter = frontmatter.rstrip() + f"\ncluster: {cluster_id}\n"

        # Add cluster tag to tags if not there
        if cluster_tag not in frontmatter:
            # Find the tags line
            lines = frontmatter.split("\n")
            new_lines = []
            for line in lines:
                if line.strip().startswith("tags:"):
                    # Add cluster tag to existing tags
                    line = line.rstrip("]").rstrip()
                    line += f", {cluster_tag}]"
                new_lines.append(line)
            frontmatter = "\n".join(new_lines)

        content = "---" + frontmatter + "---" + parts[2]

    # ── 2. Replace the Links section ──
    neighbor_data = neighbors.get(filename, [])

    links_text = "## Related Papers\n"
    for nb in neighbor_data:
        sim_pct = int(nb["similarity"] * 100)
        nb_cluster = cluster_lookup.get(nb["filename"], -1)
        same_cluster = "same cluster" if nb_cluster == cluster_id else f"cluster {nb_cluster}"
        links_text += f"- [[{nb['filename']}]] — {sim_pct}% similar ({same_cluster})\n"

    links_text += f"\n## Cluster\n"
    links_text += f"This paper belongs to **#{cluster_tag}** (cluster {cluster_id}).\n"

    # Replace existing Links section (or append if not found)
    if "## Links" in content:
        # Find where ## Links starts and replace to next ## or end
        link_start = content.index("## Links")
        # Find the next ## heading after Links
        rest = content[link_start + len("## Links"):]
        next_heading = rest.find("\n## ")
        if next_heading != -1:
            content = content[:link_start] + links_text + rest[next_heading + 1:]
        else:
            content = content[:link_start] + links_text
    elif "## Related Papers" in content:
        # Already been linked before — replace
        link_start = content.index("## Related Papers")
        rest = content[link_start + len("## Related Papers"):]
        next_heading = rest.find("\n## ")
        if next_heading != -1:
            # Find sections after Related Papers and Cluster
            remaining = rest[next_heading + 1:]
            if "## Cluster" in remaining:
                cluster_start = remaining.index("## Cluster")
                after_cluster = remaining[cluster_start + len("## Cluster"):]
                next_after = after_cluster.find("\n## ")
                if next_after != -1:
                    content = content[:link_start] + links_text + after_cluster[next_after + 1:]
                else:
                    content = content[:link_start] + links_text
            else:
                content = content[:link_start] + links_text + remaining
        else:
            content = content[:link_start] + links_text
    else:
        # No Links section exists — append
        content = content.rstrip() + "\n\n" + links_text

    note_path.write_text(content, encoding="utf-8")
    return True


def main():
    print("Loading data...")
    filenames, embeddings, cluster_lookup = load_data()
    print(f"Loaded {len(filenames)} papers\n")

    print(f"Finding {N_LINKS} nearest neighbors per paper...")
    neighbors = find_nearest_neighbors(filenames, embeddings, N_LINKS)

    print(f"Updating Obsidian notes with links and cluster tags...\n")
    updated = 0
    missing = 0

    for filename in filenames:
        note_path = NOTES_DIR / (filename + ".md")
        if update_note(note_path, filename, neighbors, cluster_lookup):
            updated += 1
        else:
            missing += 1

    print(f"{'='*40}")
    print(f"Done!")
    print(f"  Updated: {updated} notes")
    print(f"  Missing: {missing} notes")
    print(f"\nNow open Obsidian:")
    print(f"  1. Click any note — you'll see [[links]] to related papers")
    print(f"  2. Open Graph View (Cmd+G or click the graph icon)")
    print(f"  3. In Graph View settings, under 'Groups':")
    print(f"     - Add a group with query: tag:#cluster_0")
    print(f"     - Pick a color for that group")
    print(f"     - Repeat for cluster_1, cluster_2, etc.")
    print(f"     This colors your graph by cluster!")


if __name__ == "__main__":
    main()