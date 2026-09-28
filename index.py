import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATA_DIR = Path("data")
INDEX_FILE = Path("index.json")
EMBEDDING_MODEL = "text-embedding-3-small"


def split_into_chunks(text: str, max_words: int = 120, overlap: int = 20):
    """Split text into overlapping word groups."""
    words = text.split()
    chunks = []
    step = max_words - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)

    return chunks


def main():
    client = OpenAI()
    records = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")

        for chunk_number, chunk in enumerate(split_into_chunks(text), start=1):
            records.append({
                "source": path.name,
                "chunk_number": chunk_number,
                "text": chunk,
            })

    if not records:
        raise SystemExit("No .txt files found in data/")

    # Send the passages together rather than making one API call per passage.
    result = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[record["text"] for record in records],
    )

    for record, item in zip(records, result.data):
        record["embedding"] = item.embedding

    INDEX_FILE.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    print(f"Indexed {len(records)} chunks from data/ into {INDEX_FILE}")


if __name__ == "__main__":
    main()