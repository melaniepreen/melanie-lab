"""
Step 3: Extract clean text from every PDF in /pdfs/
Saves one .txt file per paper in /pdfs/extracted/
"""

import os
from pathlib import Path
from tqdm import tqdm

# ── Paths ──
PROJECT_DIR = Path(__file__).parent.parent  # melanie-lab/
PDF_DIR = PROJECT_DIR / "pdfs"
EXTRACT_DIR = PDF_DIR / "extracted"

# Create the output folder if it doesn't exist
EXTRACT_DIR.mkdir(exist_ok=True)


def extract_one_pdf(pdf_path):
    """
    Takes a single PDF file path.
    Returns the full text as a string.
    """
    from unstructured.partition.pdf import partition_pdf

    # This is the magic line — unstructured reads the PDF
    # and splits it into text elements (paragraphs, titles, etc.)
    elements = partition_pdf(str(pdf_path))

    # Join all elements into one big string
    text = "\n\n".join([str(el) for el in elements])
    return text


def main():
    # Find all PDFs
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {PDF_DIR}\n")

    if len(pdf_files) == 0:
        print("No PDFs found! Make sure they're in the /pdfs/ folder.")
        return

    # Process each one
    success = 0
    failed = []

    for pdf_path in tqdm(pdf_files, desc="Extracting"):
        txt_path = EXTRACT_DIR / (pdf_path.stem + ".txt")

        # Skip if already extracted (so you can re-run safely)
        if txt_path.exists():
            success += 1
            continue

        try:
            text = extract_one_pdf(pdf_path)

            # Only save if we actually got text
            if len(text.strip()) > 100:
                txt_path.write_text(text, encoding="utf-8")
                success += 1
            else:
                print(f"\n  Warning: very little text from {pdf_path.name}")
                failed.append(pdf_path.name)

        except Exception as e:
            print(f"\n  Error on {pdf_path.name}: {e}")
            failed.append(pdf_path.name)

    # Summary
    print(f"\n{'='*40}")
    print(f"Done! {success} extracted, {len(failed)} failed.")
    if failed:
        print(f"Failed files: {failed}")
    print(f"Text files saved to: {EXTRACT_DIR}")


if __name__ == "__main__":
    main()