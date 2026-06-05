# Melanie Lab

Personal research pipeline: **PDFs → extracted text → sentence embeddings → 2D map (UMAP) + clusters (HDBSCAN) → Obsidian notes and links**, with optional **Anthropic Claude** steps for summaries, cluster labels, and cross-paper analysis.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10+ recommended (tested with 3.12). |
| **Anthropic API** | Required for `04_visualize.py` (cluster labels), `05_obsidian_notes.py`, `07_cross_analysis.py`. Set `ANTHROPIC_API_KEY` in a `.env` file (see below). |
| **Disk / RAM** | `sentence-transformers` pulls **~90MB** for `all-MiniLM-L6-v2` on first run; PyTorch may add several GB. PDF extraction can be memory-heavy on large files. |
| **PDF tooling** | `unstructured[pdf]` may need extra system libraries on some OSes (e.g. Poppler for certain PDF paths). If extraction fails, check [Unstructured installation docs](https://docs.unstructured.io/open-source/installation/full-installation). |

### Environment variables

Create a **`.env`** file in the project root (this repo already **gitignores** `.env`):

```bash
ANTHROPIC_API_KEY=your_key_here
```

The official Anthropic Python client reads this variable by default when you call `Anthropic()` after `load_dotenv()`.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Repository layout (conceptual)

| Path | Role |
|------|------|
| `pdfs/<collection>/` | Input PDFs (e.g. `papers-meditation`, `papers-sound`). |
| `pdfs/extracted/` | One `.txt` per paper after extraction. |
| `output/` | Embeddings, cluster coordinates, HTML map (**gitignored** in this repo). |
| `notes_vault/papers/` | Per-paper Obsidian markdown. |
| `notes_vault/clusters/` | Cluster-level overview notes from cross-analysis. |
| `scripts/` | Numbered pipeline steps. |

---

## Meditation papers: folder setup and clustering

The code **does not include** your PDFs. You enable everything locally by adding files under **`pdfs/`** and running the scripts.

### 1. Create the meditation folder and add PDFs

1. From the project root, ensure this directory exists:
   - **`pdfs/papers-meditation/`**
2. Copy your **meditation-related PDFs** into **`pdfs/papers-meditation/`** (only `*.pdf` files are used by `08_process_all.py`).
3. (Optional) For the sound corpus, use **`pdfs/papers-sound/`** the same way.

These two folders are **gitignored** so your papers are not pushed to GitHub; anyone cloning the repo must repeat this setup on their machine.

### 2. Point the master script at your folders

Open **`scripts/08_process_all.py`** and check the list **`PAPER_FOLDERS`**. Each entry is a **folder name under `pdfs/`** (not the full path), for example:

```python
PAPER_FOLDERS = [
    "papers-meditation",
    "papers-sound",
]
```

- Use **only meditation:** set `PAPER_FOLDERS = ["papers-meditation"]` if you do not want sound papers in the same run.
- Add another collection: create `pdfs/<name>/`, put PDFs there, and add `"<name>"` to `PAPER_FOLDERS`.

### 3. Run extraction → embeddings → clusters

Run the master script so text is extracted, embedded, and **UMAP + HDBSCAN** run on the **combined** set of papers from all listed folders:

```bash
python3 scripts/08_process_all.py
```

That produces **`output/embeddings.json`** and **`output/corpus_map.csv`** (each row: `filename`, `x`, `y`, `cluster`, etc.). **Clusters are not “meditation-only” labels** — HDBSCAN finds dense groups in the 2D UMAP layout across **whatever papers** you included. More papers and clearer topical separation usually yield more meaningful clusters.

### 4. Finish the workflow (map, notes, links, cross-cluster writeups)

```bash
python3 scripts/04_visualize.py && \
python3 scripts/05_obsidian_notes.py && \
python3 scripts/06_link_notes.py && \
python3 scripts/07_cross_analysis.py
```

### 5. If you see almost everything “unclustered”

In **`08_process_all.py`**, HDBSCAN uses parameters such as **`min_cluster_size`** (currently **4**). A cluster must have **at least that many papers** in a dense region of the map. With very few PDFs, or very loose similarity, most points may get cluster **`-1`** (noise). Options: add more papers, or lower **`min_cluster_size`** / tune **`min_samples`** in that script (and optionally align **`scripts/03_cluster.py`** if you use it standalone).

---

## Pipeline: inputs and outputs

### Master run (`08_process_all.py`)

Configured paper folders (edit in script): `papers-meditation`, `papers-sound`.

| Step | Script / function | Input | Output |
|------|-------------------|--------|--------|
| 1 — Extract | `08_process_all.step_1_extract` | `pdfs/<folder>/*.pdf` | `pdfs/extracted/<folder>__<stem>.txt` |
| 2 — Embed | `step_2_embed` | `pdfs/extracted/*.txt` | `output/embeddings.json` |
| 3 — Cluster | `step_3_cluster` | `output/embeddings.json` | `output/corpus_map.csv` |

Then run the remaining steps in order (or use the one-liner printed at the end of `08_process_all.py`).

### Individual scripts

| Script | Input | Output |
|--------|--------|--------|
| `01_extract.py` | `pdfs/*.pdf` (flat folder) | `pdfs/extracted/<stem>.txt` |
| `02_embed.py` | `pdfs/extracted/*.txt` | `output/embeddings.json` |
| `03_cluster.py` | `output/embeddings.json` | `output/corpus_map.csv` |
| `04_visualize.py` | `output/corpus_map.csv`, extracted text for titles/snippets | `output/corpus_map.html` (+ Anthropic calls for human-readable cluster labels) |
| `05_obsidian_notes.py` | `pdfs/extracted/*.txt` | `notes_vault/papers/<stem>.md` |
| `06_link_notes.py` | `output/embeddings.json`, `output/corpus_map.csv`, existing notes | Updates notes: `[[wiki-links]]`, cluster tags in frontmatter |
| `07_cross_analysis.py` | `output/corpus_map.csv`, `notes_vault/papers/*.md` | `notes_vault/clusters/*.md` (cluster overviews via Claude) |
| `08_process_all.py` | PDFs under configured `PAPER_FOLDERS` | Extract + embed + cluster as above |

**Obsidian:** point Obsidian at `notes_vault` as the vault. The HTML map uses `obsidian://` links; adjust `OBSIDIAN_VAULT` in `04_visualize.py` if your vault name differs.

**Note:** `01_extract` vs `08` differ slightly (`01` expects PDFs directly under `pdfs/`; `08` uses subfolders and the `tradition__stem` filename prefix). Prefer **`08_process_all.py`** if you use the subfolder layout.

---

## Mathematics and methods (what UMAP is doing)

### 1. Sentence embeddings (before UMAP)

Each paper’s text is encoded into a fixed-length vector **\( \mathbf{z} \in \mathbb{R}^{384} \)** (for `all-MiniLM-L6-v2`). Semantically similar passages tend to have **small cosine distance** (large cosine similarity):

\[
d_{\cos}(\mathbf{z}_i, \mathbf{z}_j) = 1 - \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\|\mathbf{z}_i\|\,\|\mathbf{z}_j\|}.
\]

These vectors are the **high-dimensional representation** you cluster and link on.

### 2. UMAP — dimensionality reduction

**UMAP** (Uniform Manifold Approximation and Projection) maps each **384-D** embedding to **2-D coordinates** \((x, y)\) for visualization. Informally:

1. **Neighborhood graph:** For each point, UMAP considers its **\(k\)** nearest neighbors in the chosen metric (this project uses **`metric="cosine"`** on the embedding vectors, which matches angular similarity of text embeddings).
2. **Fuzzy simplicial set:** It assigns **strengths** to edges so that *local* neighborhoods are preserved in a probabilistic way (related to ideas from topological data analysis and spectral methods).
3. **Optimization in 2D:** It places points in the plane and adjusts positions to make **2D distances** reflect those fuzzy neighborhood relationships, controlled by hyperparameters such as **`n_neighbors`**, **`min_dist`** (how tight points can pack), and **`spread`**.

The 2D layout is **not** an isometry: it **compresses** 384 dimensions into 2 for human viewing, so **global distances** in the plot are interpretable only cautiously; **local neighborhoods** are what UMAP is designed to preserve best.

**Why UMAP here?** You get a single **corpus map** where nearby dots often mean “similar text,” which is easier to explore than raw vectors.

### 3. HDBSCAN — clustering (not k-means)

After UMAP, this repo runs **HDBSCAN** on the **2D** \((x, y)\) coordinates with **Euclidean** distance. HDBSCAN is **density-based**: it finds regions where many points are close together, does **not** require you to choose **k**, and can label sparse points as **noise** (**cluster id `-1`**). That differs from **k-means**, which fixes **k** centroids and assigns every point to a cluster.

### 4. Nearest-neighbor links (separate from HDBSCAN)

`06_link_notes.py` builds **Obsidian `[[links]]`** using **cosine similarity on the original 384-D embeddings**, not k-means. So:

- **Cluster label** ≈ dense group in the **2D UMAP** picture (HDBSCAN).
- **`[[related paper]]` links** ≈ **top‑N most similar** papers in **embedding space**.

Those two notions are related but not identical.

---

## Security and publishing to GitHub

This section is for **public** or **shared** repositories.

### What this repo already ignores (good)

From `.gitignore`:

- **`.env`** — prevents committing API keys.
- **`output/`** — avoids committing **`embeddings.json`** (dense vectors + filenames) and generated CSV/HTML.
- **`pdfs/papers-meditation/`**, **`pdfs/papers-sound/`**, and matching **`pdfs/extracted/papers-meditation__*.txt`** / **`papers-sound__*.txt`** — local PDF corpora and their extracted text are not committed.

### Risks to review before you push

| Risk | Mitigation |
|------|------------|
| **API keys** | Never commit `.env` or keys in code. Rotate any key that was ever committed. |
| **`pdfs/`** | **Meditation and sound** subfolders and their extracted `*.txt` are gitignored. Any **other** paths under `pdfs/` are still tracked if you add them — PDFs may be **copyrighted**; use care before pushing. |
| **`notes_vault/`** | Not ignored by default. Notes contain **Claude-generated text** and possibly **snippets derived from papers** — check **licensing** and **privacy** before publishing. |
| **Third-party API** | Scripts send **paper text** (truncated chunks) and **titles** to **Anthropic**. Review [Anthropic’s data use / retention policies](https://www.anthropic.com/) for your compliance needs. |
| **Embeddings** | If `output/` were ever committed, vectors + filenames can **reveal structure** of your corpus; keep `output/` private or gitignored. |
| **Dependency supply chain** | Pin versions in production; run **`pip audit`** periodically on `requirements.txt`. |

### Suggested checks before going public

```bash
git status
git diff --cached
# Ensure .env, output/, and any local secrets are not staged
```

Use [GitHub secret scanning](https://docs.github.com/en/code-security/secret-scanning) and avoid storing keys in issue text or README examples (use placeholders only).

---

## Quick start (subfolder layout + full flow)

1. Put PDFs in **`pdfs/papers-meditation/`** (and optionally **`pdfs/papers-sound/`**), matching **`PAPER_FOLDERS`** in `scripts/08_process_all.py`. See **[Meditation papers: folder setup and clustering](#meditation-papers-folder-setup-and-clustering)** above.
2. `python3 scripts/08_process_all.py`
3. Then:

   ```bash
   python3 scripts/04_visualize.py && \
   python3 scripts/05_obsidian_notes.py && \
   python3 scripts/06_link_notes.py && \
   python3 scripts/07_cross_analysis.py
   ```

4. Open `output/corpus_map.html` in a browser and `notes_vault` in Obsidian.

---

## License

The **code in this repository** is licensed under the [MIT License](LICENSE).

**PDFs and generated notes** are separate from that license: copyright stays with publishers/authors. The **meditation** and **sound** corpora are **gitignored** (`pdfs/papers-meditation/`, `pdfs/papers-sound/`, and matching `pdfs/extracted/papers-meditation__*.txt` / `papers-sound__*.txt`) so they are not pushed to GitHub. Any other paths under `pdfs/` or `notes_vault/` remain your responsibility if you make the repo public.
