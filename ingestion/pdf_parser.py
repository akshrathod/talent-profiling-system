"""
Extracts clean text from PDFs in data/raw/ and saves them
as .txt files in data/parsed/. Also uploads parsed files to S3.
"""

import fitz
import boto3
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET     = os.getenv("S3_BUCKET_NAME", "talent-profiling-raw-docs")  # default if not saved in .env file
LOCAL_RAW_DIR  = Path("data/raw")
LOCAL_PARSED_DIR = Path("data/parsed")


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "us-east-1"),
    )


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract and clean text from a single PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []

    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())

    doc.close()
    full_text = "\n\n".join(pages)
    return full_text


def clean_text(text: str) -> str:
    """Basic cleaning - remove excessive whitespace and blank lines."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


def enrich_with_metadata(text: str, registry_entry: dict) -> str:
    """
    Prepend paper metadata to the extracted text.
    This gives the LLM structured context before the raw paper body.
    """
    authors = ", ".join(registry_entry.get("authors", [])[:10])
    categories = ", ".join(registry_entry.get("categories", []))
    abstract = registry_entry.get("abstract", "").replace("\n", " ")

    header = f"""TITLE: {registry_entry.get('title', 'Unknown')}
AUTHORS: {authors}
CATEGORIES: {categories}
PUBLISHED: {registry_entry.get('published', 'Unknown')}
ABSTRACT: {abstract}

FULL TEXT:
"""
    return header + text


def parse_all(upload_to_s3: bool = True):
    LOCAL_PARSED_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client() if upload_to_s3 else None

    registry_path = LOCAL_RAW_DIR / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        registry_map = {Path(r["local_path"]).name: r for r in registry}
    else:
        registry_map = {}

    pdfs = list(LOCAL_RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs to parse\n")

    success_count = 0
    fail_count = 0

    for pdf_path in pdfs:
        txt_filename = pdf_path.stem + ".txt"
        txt_path = LOCAL_PARSED_DIR / txt_filename

        if txt_path.exists():
            print(f"Skipping (already parsed): {pdf_path.name}")
            continue

        print(f"Parsing: {pdf_path.name}")

        try:
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)

            registry_entry = registry_map.get(pdf_path.name, {})
            if registry_entry:
                final_text = enrich_with_metadata(cleaned, registry_entry)
            else:
                final_text = cleaned

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_text)

            print(f"  Saved: {txt_path}")

            if upload_to_s3 and s3:
                s3_key = f"parsed/{txt_filename}"
                s3.upload_file(str(txt_path), S3_BUCKET, s3_key)
                print(f"  Uploaded to S3: {s3_key}")

            success_count += 1

        except Exception as e:
            print(f"  Failed: {e}")
            fail_count += 1

    print(f"\nDone. Success: {success_count}  Failed: {fail_count}")
    return success_count, fail_count


if __name__ == "__main__":
    parse_all()