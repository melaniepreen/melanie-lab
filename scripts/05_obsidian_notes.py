"""
Step 4.5: Generate Obsidian notes for each paper using Claude.
Each note contains structured frontmatter (Layer 4 fields)
and a Claude-generated summary.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from tqdm import tqdm

# ── Config ──
load_dotenv()
PROJECT_DIR = Path(__file__).parent.parent
EXTRACT_DIR = PROJECT_DIR / "pdfs" / "extracted"
NOTES_DIR = PROJECT_DIR / "notes_vault" / "papers"

# Create the papers subfolder in your vault
NOTES_DIR.mkdir(parents=True, exist_ok=True)

client = Anthropic()


def read_paper_text(txt_path, max_chars=6000):
    """
    Reads extracted text, truncated to fit Claude's context
    while capturing the most important parts.
    """
    text = txt_path.read_text(encoding="utf-8")

    # If short enough, use it all
    if len(text) <= max_chars:
        return text

    # Otherwise: take first 4000 (abstract + intro) + last 2000 (conclusion)
    return text[:4000] + "\n\n[...middle truncated...]\n\n" + text[-2000:]


def generate_note(filename, text):
    """
    Sends paper text to Claude and gets back a structured
    Obsidian note with YAML frontmatter + summary.
    """
    prompt = f"""You are a research assistant for an integrative medicine lab.
Read the following academic paper text and produce a structured Obsidian 
markdown note. Follow this EXACT format — no deviation:

---
title: "[paper title]"
authors: [First Author, Second Author]
year: [publication year, or "unknown"]
journal: "[journal name, or unknown]"
tradition: [e.g. mindfulness, meditation, vipassana, zen, MBSR, contemplative, or general]
study_type: [RCT, observational, review, meta-analysis, neuroimaging, qualitative, or other]
condition: "[what condition or outcome is studied, e.g. anxiety, attention, pain]"
intervention: "[what intervention is tested, e.g. 8-week MBSR program]"
sample_size: [number, or unknown]
effect_direction: [positive, negative, mixed, or unclear]
tags: [tag1, tag2, tag3]
---

# [Paper Title]

## Summary
[3-4 sentence summary of the key finding and methodology]

## Proposed Mechanism
[1-2 sentences on how/why the intervention might work. If no mechanism is discussed, write "Not discussed in this paper."]

## Key Findings
[2-3 bullet points of the most important results]

## Limitations
[1-2 sentences on study limitations mentioned by the authors]

## Mechanism Gap
[Does this paper leave any mechanism unexplained? If yes, describe briefly. If no, write "No significant mechanism gap identified."]

## Links
[Leave empty for now — will be populated as more papers are indexed]

IMPORTANT RULES:
- Use ONLY information from the paper text below. Never invent data.
- If you cannot determine a field, write "unknown" — never guess.
- Keep the summary concise and factual.
- The YAML frontmatter must be valid YAML.

PAPER TEXT:
{text}"""

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text


def main():
    # Find all extracted text files
    txt_files = sorted(EXTRACT_DIR.glob("*.txt"))
    print(f"Found {len(txt_files)} extracted papers")
    print(f"Notes will be saved to: {NOTES_DIR}\n")

    if len(txt_files) == 0:
        print("No text files found! Run 01_extract.py first.")
        return

    success = 0
    failed = []
    skipped = 0

    for txt_path in tqdm(txt_files, desc="Generating notes"):
        note_path = NOTES_DIR / (txt_path.stem + ".md")

        # Skip if note already exists (safe to re-run)
        if note_path.exists():
            skipped += 1
            continue

        try:
            text = read_paper_text(txt_path)
            note_content = generate_note(txt_path.stem, text)

            # Save the note
            note_path.write_text(note_content, encoding="utf-8")
            success += 1

            # Small delay to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            print(f"\n  Error on {txt_path.stem}: {e}")
            failed.append(txt_path.stem)
            # Longer delay on error (might be rate limit)
            time.sleep(5)

    print(f"\n{'='*40}")
    print(f"Done!")
    print(f"  Generated: {success}")
    print(f"  Skipped (already existed): {skipped}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed files: {failed}")
    print(f"\nNotes saved to: {NOTES_DIR}")
    print(f"\nOpen Obsidian — your notes should appear in the 'papers' folder.")


if __name__ == "__main__":
    main()