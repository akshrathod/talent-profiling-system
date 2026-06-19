"""
Downloads real research papers from arXiv across multiple ML/AI domains.
Saves PDFs to data/raw/ locally AND uploads them to S3 bucket.
"""

import arxiv
import boto3
import os
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Config

S3_BUCKET       = os.getenv("S3_BUCKET_NAME", "talent-profiling-raw-docs")  # saved in .env file
LOCAL_RAW_DIR   = Path("data/raw")
PAPERS_PER_TOPIC = 10         # 5 topics x 10 papers = 50 papers total

SEARCH_TOPICS = [
    "large language models transformer fine-tuning",
    "reinforcement learning policy gradient reward",
    "graph neural networks node classification",
    "machine learning systems MLOps deployment",
    "AI safety alignment robustness",
]

# S3 Client

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name           = os.getenv("AWS_REGION", "us-east-1"),
    )

# Helpers

def sanitize_filename(title: str) -> str:
    """Turn a paper title into a safe filename."""
    keep = set("abcdefghijklmnopqrstuvwxyz0123456789_- ")
    cleaned = "".join(c if c.lower() in keep else "_" for c in title)
    return "_".join(cleaned.split())[:80]          # max 80 chars


def download_pdf(paper, dest_path: Path) -> bool:
    """Download a single arXiv PDF. Returns True on success."""
    try:
        urllib.request.urlretrieve(paper.pdf_url, dest_path)
        return True
    except Exception as e:
        print(f"    X Download failed: {e}")
        return False


def upload_to_s3(s3, local_path: Path, s3_key: str) -> bool:
    """Upload a local file to S3. Returns True on success."""
    try:
        s3.upload_file(str(local_path), S3_BUCKET, s3_key)
        return True
    except Exception as e:
        print(f"    X S3 upload failed: {e}")
        return False

# Core ingestion

def ingest_papers():
    LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()

    total_downloaded = 0
    total_uploaded   = 0
    registry         = []          # track what we collected

    print(f"\n{'='*60}")
    print(f"  arXiv Ingester - {len(SEARCH_TOPICS)} topics x {PAPERS_PER_TOPIC} papers")
    print(f"  Local -> {LOCAL_RAW_DIR.resolve()}")
    print(f"  Cloud -> s3://{S3_BUCKET}/raw/")
    print(f"{'='*60}\n")

    for topic in SEARCH_TOPICS:
        print(f"Topic: {topic}")

        client = arxiv.Client()
        search = arxiv.Search(
            query      = topic,
            max_results= PAPERS_PER_TOPIC,
            sort_by    = arxiv.SortCriterion.Relevance,
        )

        for paper in client.results(search):
            filename  = sanitize_filename(paper.title) + ".pdf"
            local_path = LOCAL_RAW_DIR / filename
            s3_key    = f"raw/{filename}"

            print(f"  -> {paper.title[:60]}...")

            # Skip if already downloaded
            if local_path.exists():
                print(f"    Already exists locally, skipping download")
            else:
                success = download_pdf(paper, local_path)
                if success:
                    total_downloaded += 1
                    print(f"    Downloaded")
                else:
                    continue

            # Upload to S3
            if upload_to_s3(s3, local_path, s3_key):
                total_uploaded += 1
                print(f"    Uploaded to S3")

            # Store metadata in registry
            registry.append({
                "title"      : paper.title,
                "authors"    : [a.name for a in paper.authors],
                "abstract"   : paper.summary,
                "categories" : paper.categories,
                "published"  : str(paper.published),
                "pdf_url"    : paper.pdf_url,
                "local_path" : str(local_path),
                "s3_key"     : s3_key,
            })

            time.sleep(0.5)          # be polite to arXiv API

        print()

    # Save registry
    import json
    registry_path = LOCAL_RAW_DIR / "registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # Upload registry to S3 too
    upload_to_s3(s3, registry_path, "raw/registry.json")

    print(f"{'='*60}")
    print(f"  Downloaded : {total_downloaded} PDFs")
    print(f"  Uploaded   : {total_uploaded} files to S3")
    print(f"  Registry   : {registry_path}")
    print(f"{'='*60}\n")

    return registry


# Entry point

if __name__ == "__main__":
    registry = ingest_papers()
    print(f"Done. {len(registry)} papers ready for parsing.")
