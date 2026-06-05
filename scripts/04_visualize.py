"""
Step 6: Build the interactive corpus map.
Reads corpus_map.csv, generates cluster labels with Claude,
outputs an HTML file you can open in any browser.
Clicking a dot opens the paper's note in Obsidian.
"""

import os
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

# ── Config ──
load_dotenv()
PROJECT_DIR = Path(__file__).parent.parent
CSV_FILE = PROJECT_DIR / "output" / "corpus_map.csv"
EXTRACT_DIR = PROJECT_DIR / "pdfs" / "extracted"
OUTPUT_HTML = PROJECT_DIR / "output" / "corpus_map.html"

# ── CHANGE THIS to your vault name if different ──
OBSIDIAN_VAULT = "notes_vault"


def get_paper_title(filename):
    """
    Reads the first few lines of the extracted text
    to grab a rough title. Better than just the filename.
    """
    txt_path = EXTRACT_DIR / (filename + ".txt")
    if not txt_path.exists():
        return filename

    text = txt_path.read_text(encoding="utf-8")
    # First non-empty line is usually the title
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 10 and len(line) < 300:
            return line[:120]
    return filename


def get_paper_snippet(filename, max_chars=300):
    """
    Grabs the first ~300 chars of the paper for the hover tooltip.
    """
    txt_path = EXTRACT_DIR / (filename + ".txt")
    if not txt_path.exists():
        return ""

    text = txt_path.read_text(encoding="utf-8")
    # Skip very short lines (headers, page numbers) to find real content
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    content = " ".join(lines[:5])
    return content[:max_chars] + "..."


def label_clusters_with_claude(df):
    """
    Sends each cluster's paper titles to Claude and asks
    for a short, descriptive label.
    """
    client = Anthropic()
    cluster_labels = {}

    unique_clusters = sorted([c for c in df["cluster"].unique() if c != -1])

    print(f"Asking Claude to label {len(unique_clusters)} clusters...\n")

    for cluster_id in unique_clusters:
        # Get all paper titles in this cluster
        cluster_papers = df[df["cluster"] == cluster_id]
        titles = [get_paper_title(f) for f in cluster_papers["filename"]]
        titles_text = "\n".join([f"- {t}" for t in titles])

        message = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"""Here are {len(titles)} research paper titles from one cluster. 
Give me a short label (2-5 words) that describes what these papers have in common. 
Reply with ONLY the label, nothing else.

Papers:
{titles_text}"""
            }]
        )

        label = message.content[0].text.strip().strip('"').strip("'")
        cluster_labels[cluster_id] = label
        print(f"  Cluster {cluster_id} ({len(titles)} papers) → {label}")

    # Unclustered papers get a default label
    cluster_labels[-1] = "Unclustered"

    return cluster_labels


def build_obsidian_url(filename):
    """
    Creates an obsidian:// URL that opens the paper's note.
    """
    note_name = f"papers/{filename}"
    return f"obsidian://open?vault={OBSIDIAN_VAULT}&file={note_name}"


def build_map(df, cluster_labels):
    tradition_shapes = {
        "papers-meditation": "circle",
        "papers-sound": "star",
        "ayurveda": "diamond",
        "tcm": "triangle-up",
        "western-herbal": "square",
        "quantum-biology": "hexagon",
        "mind-body": "pentagon",
        "unknown": "circle",
    }
    """
    Creates the interactive Plotly map.
    """
    # Color palette — distinct, colorblind-friendly
    colors = [
        "#2D6A4F", "#E76F51", "#264653", "#E9C46A",
        "#F4A261", "#606C38", "#9B2226", "#457B9D",
        "#6D597A", "#B5838D", "#3D405B", "#81B29A"
    ]

    fig = go.Figure()

    # Plot each cluster as its own trace (so legend works)
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_df = df[df["cluster"] == cluster_id]
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")

        if cluster_id == -1:
            color = "#CCCCCC"
            opacity = 0.5
        else:
            color = colors[cluster_id % len(colors)]
            opacity = 0.85

        # Build hover text for each paper
# Build hover text for each paper
        hover_texts = []
        obsidian_urls = []
        for _, row in cluster_df.iterrows():
            title = get_paper_title(row["filename"])
            # Wrap title to max 60 chars per line
            words = title.split()
            title_lines = []
            current_line = ""
            for word in words:
                if len(current_line + " " + word) > 60:
                    title_lines.append(current_line)
                    current_line = word
                else:
                    current_line = (current_line + " " + word).strip()
            if current_line:
                title_lines.append(current_line)
            wrapped_title = "<br>".join(title_lines)
            
            tradition = row.get("tradition", "unknown")
            tradition_display = tradition.replace("papers-", "").replace("-", " ").title()
            hover = (
                f"<b>{wrapped_title}</b><br>"
                f"<i>{label}</i><br>"
                f"Tradition: {tradition_display}<br>"
                f"<br>"
                f"Click to open in Obsidian"
            )
            hover_texts.append(hover)
            obsidian_urls.append(build_obsidian_url(row["filename"]))

        fig.add_trace(go.Scatter(
            x=cluster_df["x"],
            y=cluster_df["y"],
            mode="markers",
            name=label,
# Determine shapes per point based on tradition
            point_symbols = []
            for _, row in cluster_df.iterrows():
                tradition = row.get("tradition", "unknown")
                point_symbols.append(tradition_shapes.get(tradition, "circle"))

            marker=dict(
                size=18,
                color=color,
                symbol=point_symbols,
                opacity=opacity,
                line=dict(width=1.5, color="white")
            ),
            hovertext=hover_texts,
            hoverinfo="text",
            customdata=obsidian_urls,
        ))

    # ── Layout styling ──
    fig.update_layout(
        title=dict(
            text="Melanie Lab — Corpus Map",
            font=dict(size=24, family="Helvetica Neue, Arial", color="#1a1a2e"),
            x=0.5
        ),
        plot_bgcolor="#FAFAF5",
        paper_bgcolor="#FAFAF5",
        font=dict(family="Helvetica Neue, Arial", size=12, color="#333"),
        legend=dict(
            title="Clusters",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ddd",
            borderwidth=1,
            font=dict(size=11)
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#333",
            font_size=13,
            font_family="Helvetica Neue, Arial",
            font_color="#1a1a2e",
            align="left",
            namelength=-1
        ),
        xaxis=dict(
            showgrid=False, showticklabels=False,
            zeroline=False, title=""
        ),
        yaxis=dict(
            showgrid=False, showticklabels=False,
            zeroline=False, title=""
        ),
        width=1200,
        height=800,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # ── Add click-to-Obsidian behavior via JavaScript ──
    fig.write_html(
        str(OUTPUT_HTML),
        include_plotlyjs=True,
        post_script="""
        var plot = document.getElementsByClassName('plotly')[0];
        plot.on('plotly_click', function(data) {
            var url = data.points[0].customdata;
            if (url) {
                window.open(url, '_blank');
            }
        });
        """
    )


def main():
    # Load the data
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} papers from {CSV_FILE}\n")

    # Get cluster labels from Claude
    cluster_labels = label_clusters_with_claude(df)

    # Build the map
    print(f"\nBuilding map...")
    build_map(df, cluster_labels)

    print(f"\n{'='*40}")
    print(f"Done! Map saved to: {OUTPUT_HTML}")
    print(f"\nOpen it with:")
    print(f"  open {OUTPUT_HTML}")


if __name__ == "__main__":
    main()