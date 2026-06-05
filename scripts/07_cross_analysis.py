"""
Cross-paper analysis: reads ALL notes in a cluster,
sends them to Claude together, and generates:
  1. Cluster summary (what this group of papers tells us collectively)
  2. Key findings (the strongest evidence across papers)
  3. Mechanism gaps (what's unexplained)
  4. Contradictions (where papers disagree)
  5. Hypotheses (testable ideas that emerge from the cluster)

Saves one "cluster overview" note per cluster in Obsidian.
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
PROJECT_DIR = Path(__file__).parent.parent
NOTES_DIR = PROJECT_DIR / "notes_vault" / "papers"
OVERVIEW_DIR = PROJECT_DIR / "notes_vault" / "clusters"
CLUSTER_FILE = PROJECT_DIR / "output" / "corpus_map.csv"

OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

client = Anthropic()


def read_all_notes_in_cluster(cluster_id, df):
    """
    Reads every Obsidian note for papers in this cluster.
    Returns a combined text block with clear separators.
    """
    cluster_papers = df[df["cluster"] == cluster_id]
    combined = []

    for _, row in cluster_papers.iterrows():
        note_path = NOTES_DIR / (row["filename"] + ".md")
        if note_path.exists():
            content = note_path.read_text(encoding="utf-8")
            combined.append(f"=== PAPER: {row['filename']} ===\n{content}")

    return "\n\n".join(combined)


def generate_cluster_overview(cluster_id, cluster_text, paper_count):
    """
    Sends all papers in a cluster to Claude for cross-paper analysis.
    """
    prompt = f"""You are a senior researcher at an integrative medicine lab.
Below are {paper_count} research papers that have been automatically clustered
together because they are semantically similar. They all relate to meditation
or contemplative practice research.

Read ALL of them carefully, then produce a cluster overview note in this
EXACT format:

---
type: cluster_overview
cluster_id: {cluster_id}
paper_count: {paper_count}
---

# Cluster {cluster_id} Overview

## Collective Summary
[4-6 sentences synthesizing what this group of papers tells us AS A WHOLE.
Not a list of individual summaries — a genuine synthesis of the collective
knowledge. What picture emerges when you read them together?]

## Strongest Evidence
[The 3-5 most robust findings across all papers in this cluster.
For each finding, note which paper(s) support it and the study quality.
Format as bullet points:]
- **[Finding]** — supported by [paper name(s)], [study type], n=[sample size]

## Mechanism Gaps
[What mechanisms are proposed but poorly supported? What effects are
documented but unexplained? These are the research frontier — the
questions this cluster raises but cannot answer.]
- **[Gap 1]**: [description]
- **[Gap 2]**: [description]

## Contradictions
[Where do papers in this cluster disagree? Note the specific disagreement
and which papers are on each side. If no contradictions, write "No
significant contradictions identified."]

## Cross-Paper Hypotheses
[Based on reading these papers together, what NEW testable hypotheses
emerge? These should be ideas that no single paper states, but that
become apparent when you see the full picture. These are the most
valuable output of this analysis.]
- **Hypothesis 1**: [statement] — *suggested by*: [which papers hint at this]
- **Hypothesis 2**: [statement] — *suggested by*: [which papers hint at this]

## Papers in This Cluster
[List every paper with a wiki-link]

IMPORTANT RULES:
- Synthesize, don't summarize individual papers.
- Hypotheses must be NOVEL — not restating what papers already conclude.
- Mechanism gaps are the most valuable section. Be specific.
- Use [[wiki-links]] when referencing paper filenames.

PAPERS:
{cluster_text}"""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def generate_master_overview(cluster_overviews):
    """
    Reads ALL cluster overviews and generates a master synthesis
    that looks across the entire corpus.
    """
    combined = "\n\n".join([
        f"=== CLUSTER {cid} ===\n{text}"
        for cid, text in cluster_overviews.items()
    ])

    prompt = f"""You are the lead researcher at an integrative medicine lab
studying meditation and contemplative practices. Below are overview notes
from {len(cluster_overviews)} research clusters in your corpus.

Read ALL cluster overviews, then produce a MASTER SYNTHESIS note:

---
type: master_overview
total_clusters: {len(cluster_overviews)}
---

# Corpus Master Overview — Meditation Research

## The Big Picture
[5-8 sentences. What does your entire corpus tell you about the state
of meditation research? What's well-established, what's emerging,
what's missing?]

## Top 5 Mechanism Gaps Across All Clusters
[The most important unanswered questions across your entire corpus.
These are your research priorities. Rank by importance.]
1. **[Gap]**: [why it matters] — appears in clusters [X, Y]
2. ...

## Top 5 Testable Hypotheses
[The most promising cross-cluster hypotheses. Ideas that emerge
from seeing the FULL picture, not any single cluster.]
1. **[Hypothesis]**: [statement] — *bridges clusters*: [X and Y]
2. ...

## Evidence Landscape
[A brief assessment of evidence quality across your corpus.
How many RCTs? How many observational? What's the overall
methodological strength? Where is evidence weakest?]

## Recommended Next Papers to Index
[Based on the gaps and hypotheses above, what specific types of
papers should the lab prioritize finding and indexing next?]

## Connections to Integrative Medicine
[How do these meditation findings connect to the broader mission
of integrative medicine? Any bridges to traditional medical systems
like Ayurveda or TCM?]

CLUSTER OVERVIEWS:
{combined}"""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def main():
    df = pd.read_csv(CLUSTER_FILE)
    clusters = sorted([c for c in df["cluster"].unique() if c != -1])

    print(f"Found {len(clusters)} clusters\n")

    # ── Generate cluster overviews ──
    cluster_overviews = {}

    for cluster_id in clusters:
        paper_count = len(df[df["cluster"] == cluster_id])
        print(f"Analyzing cluster {cluster_id} ({paper_count} papers)...")

        cluster_text = read_all_notes_in_cluster(cluster_id, df)

        # Check if text is too long (Claude's context limit)
        if len(cluster_text) > 150000:
            print(f"  Warning: cluster {cluster_id} text very long, truncating...")
            cluster_text = cluster_text[:150000]

        overview = generate_cluster_overview(cluster_id, cluster_text, paper_count)
        cluster_overviews[cluster_id] = overview

        # Save cluster overview note
        overview_path = OVERVIEW_DIR / f"cluster_{cluster_id}_overview.md"
        overview_path.write_text(overview, encoding="utf-8")
        print(f"  Saved: {overview_path.name}")

    # ── Generate master overview ──
    print(f"\nGenerating master synthesis across all clusters...")
    master = generate_master_overview(cluster_overviews)

    master_path = PROJECT_DIR / "notes_vault" / "MASTER_OVERVIEW.md"
    master_path.write_text(master, encoding="utf-8")
    print(f"Saved: MASTER_OVERVIEW.md")

    print(f"\n{'='*40}")
    print(f"Done!")
    print(f"  {len(clusters)} cluster overviews → notes_vault/clusters/")
    print(f"  1 master overview → notes_vault/MASTER_OVERVIEW.md")
    print(f"\nOpen Obsidian and check:")
    print(f"  • /clusters/ folder for per-cluster analysis")
    print(f"  • MASTER_OVERVIEW.md for the full synthesis")
    print(f"  • Top mechanism gaps = your research priorities")
    print(f"  • Top hypotheses = your hypothesis feed v0.1")


if __name__ == "__main__":
    main()