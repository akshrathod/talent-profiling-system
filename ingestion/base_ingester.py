"""
Single entry point for all document ingestion.
Accepts a local file, a folder, or an S3 prefix.
Routes to the correct parser based on file extension.
Returns a list of ingested document dicts ready for the pipeline.
"""

import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET        = os.getenv("S3_BUCKET_NAME", "talent-profiling-raw-docs")
LOCAL_PARSED_DIR = Path("data/parsed")


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "us-east-1"),
    )


def _load_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_by_extension(file_path: Path) -> str:
    """Route file to the correct parser based on extension."""
    ext = file_path.suffix.lower()

    if ext == ".txt":
        from ingestion.text_parser import extract_text
        return extract_text(file_path)

    elif ext == ".pdf":
        from ingestion.pdf_parser import extract_text_from_pdf, clean_text
        raw = extract_text_from_pdf(file_path)
        return clean_text(raw)

    elif ext in (".docx", ".doc"):
        from ingestion.docx_parser import extract_text
        return extract_text(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _build_doc(file_path: Path, text: str, source: str) -> dict:
    """Wrap extracted text in a standard document dict."""
    return {
        "doc_id"   : file_path.stem,
        "filename" : file_path.name,
        "source"   : source,
        "text"     : text,
        "char_count": len(text),
    }


def ingest_from_file(file_path: str) -> list[dict]:
    """Ingest a single local file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = _parse_by_extension(path)
    doc  = _build_doc(path, text, source="local_file")
    print(f"Ingested: {path.name} ({doc['char_count']} chars)")
    return [doc]


def ingest_from_folder(folder_path: str) -> list[dict]:
    """Ingest all supported files from a local folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    supported = {".pdf", ".txt", ".docx", ".doc"}
    files     = [f for f in folder.iterdir() if f.suffix.lower() in supported]

    if not files:
        raise ValueError(f"No supported files found in: {folder_path}")

    print(f"Found {len(files)} files in {folder_path}")

    docs = []
    for file_path in files:
        try:
            text = _parse_by_extension(file_path)
            doc  = _build_doc(file_path, text, source="local_folder")
            docs.append(doc)
            print(f"  Ingested: {file_path.name} ({doc['char_count']} chars)")
        except Exception as e:
            print(f"  Failed: {file_path.name} -> {e}")

    return docs


def ingest_from_parsed(limit: int = None) -> list[dict]:
    """
    Load already-parsed .txt files from data/parsed/.
    This is the default mode for the pipeline since
    pdf_parser.py already did the heavy lifting.
    """
    txt_files = sorted(LOCAL_PARSED_DIR.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {LOCAL_PARSED_DIR}")

    if limit:
        txt_files = txt_files[:limit]

    print(f"Loading {len(txt_files)} parsed documents from {LOCAL_PARSED_DIR}")

    docs = []
    for path in txt_files:
        try:
            text = _load_txt(path)
            doc  = _build_doc(path, text, source="parsed_cache")
            docs.append(doc)
        except Exception as e:
            print(f"  Failed: {path.name} -> {e}")

    print(f"Loaded {len(docs)} documents\n")
    return docs


def ingest_from_s3(prefix: str = "parsed/", limit: int = None) -> list[dict]:
    """
    Pull parsed .txt files directly from S3.
    Used when running the pipeline in a cloud environment.
    """
    s3   = get_s3_client()
    docs = []

    paginator = s3.get_paginator("list_objects_v2")
    pages     = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)

    keys = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".txt"):
                keys.append(obj["Key"])

    if limit:
        keys = keys[:limit]

    print(f"Found {len(keys)} parsed files in s3://{S3_BUCKET}/{prefix}")

    for key in keys:
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            text     = response["Body"].read().decode("utf-8")
            filename = Path(key).name
            doc      = _build_doc(Path(filename), text, source="s3")
            docs.append(doc)
            print(f"  Loaded from S3: {filename} ({doc['char_count']} chars)")
        except Exception as e:
            print(f"  Failed: {key} -> {e}")

    print(f"Loaded {len(docs)} documents from S3\n")
    return docs


def ingest(source: str = "parsed", path: str = None, limit: int = None) -> list[dict]:
    """
    Main entry point. Call this from anywhere in the pipeline.

    Usage:
        ingest("parsed")               # load from data/parsed/ (default)
        ingest("s3")                   # load from S3 parsed/ prefix
        ingest("file", "resume.pdf")   # single file
        ingest("folder", "data/raw")   # entire folder
    """
    if source == "parsed":
        return ingest_from_parsed(limit=limit)
    elif source == "s3":
        return ingest_from_s3(limit=limit)
    elif source == "file":
        return ingest_from_file(path)
    elif source == "folder":
        return ingest_from_folder(path)
    else:
        raise ValueError(f"Unknown source: {source}. Use: parsed, s3, file, folder")


if __name__ == "__main__":
    docs = ingest("parsed", limit=3)
    for doc in docs:
        print(f"\ndoc_id    : {doc['doc_id']}")
        print(f"filename  : {doc['filename']}")
        print(f"source    : {doc['source']}")
        print(f"char_count: {doc['char_count']}")
        print(f"preview   : {doc['text'][:200]}")