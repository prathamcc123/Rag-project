"""
extract_text.py

Purpose:
Extract text from all PDF files in data/raw_pdfs/
and save them as .txt files in data/extracted_text/

Author: Major Project
"""

from pathlib import Path
from pypdf import PdfReader


# ==========================
# Folder Configuration
# ==========================

PDF_FOLDER = Path("data/raw_pdfs")
OUTPUT_FOLDER = Path("data/extracted_text")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================
# Main Function
# ==========================

def extract_text_from_pdf(pdf_path: Path):

    print("=" * 60)
    print(f"Processing PDF : {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))

        print(f"Pages Found   : {len(reader.pages)}")

        extracted_text = ""

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                page_text = page.extract_text()

                if page_text:
                    extracted_text += f"\n\n===== PAGE {page_number} =====\n\n"
                    extracted_text += page_text

            except Exception as page_error:
                print(f"⚠️ Could not read Page {page_number}: {page_error}")

        output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

        with open(output_file, "w", encoding="utf-8") as file:
            file.write(extracted_text)

        print(f"Saved Text    : {output_file.name}")
        print("Status        : SUCCESS")

    except Exception as pdf_error:

        print(f"FAILED: {pdf_path.name}")
        print(pdf_error)


# ==========================
# Entry Point
# ==========================

def main():

    print("\n========== PDF TEXT EXTRACTION ==========\n")

    print("Current Directory :", Path.cwd())
    print("PDF Folder        :", PDF_FOLDER.resolve())

    if not PDF_FOLDER.exists():
        print("\nERROR: raw_pdfs folder not found.")
        return

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    print(f"\nPDFs Detected : {len(pdf_files)}")

    if len(pdf_files) == 0:
        print("\nNo PDF files found.")
        return

    print("\nFiles Found:")

    for pdf in pdf_files:
        print(f" • {pdf.name}")

    print()

    for pdf in pdf_files:
        extract_text_from_pdf(pdf)

    print("\n" + "=" * 60)
    print("ALL PDF FILES PROCESSED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()