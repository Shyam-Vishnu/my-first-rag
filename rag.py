import math

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"


def split_into_chunks(text: str, max_words: int = 120, overlap: int = 20):
    words = text.split()
    step = max_words - overlap

    return [
        " ".join(words[start : start + max_words])
        for start in range(0, len(words), step)
        if words[start : start + max_words]
    ]


def index_documents(documents: list[tuple[str, str]]) -> list[dict]:
    """Turn (filename, extracted_text) pairs into searchable records."""
    records = []

    for filename, text in documents:
        for chunk_number, chunk in enumerate(
            split_into_chunks(text), start=1
        ):
            records.append({
                "source": filename,
                "chunk_number": chunk_number,
                "text": chunk,
            })

    if not records:
        return []

    client = OpenAI()

    # Batch calls so we don't make a separate request for every chunk.
    for start in range(0, len(records), 100):
        batch = records[start : start + 100]

        result = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[record["text"] for record in batch],
        )

        for record, item in zip(batch, result.data):
            record["embedding"] = item.embedding

    return records


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (length_a * length_b)


def make_search_question(question, history):
    if not history:
        return question

    client = OpenAI()

    response = client.responses.create(
        model=ANSWER_MODEL,
        instructions=(
            "Rewrite the latest question as a standalone search query. "
            "Use the conversation only to resolve references such as "
            "'it', 'that', or 'the same metric'. "
            "Preserve the user's intent and explicit details. "
            "Do not answer the question or invent missing details. "
            "If the reference is unclear, preserve the original question. "
            "Return only the search query."
        ),
        input=(
            f"Earlier conversation:\n{history}\n\n"
            f"Latest question:\n{question}"
        ),
        max_output_tokens=300,
        temperature=0.2,
    )

    return response.output_text.strip() or question


def ask(question, records, top_k=3, history=None):
    if not records:
        raise ValueError("Attach a document first.")

    client = OpenAI()
    recent_messages = (history or [])[-6:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_messages
    )

    search_question = make_search_question(
        question,
        history_text,
    )


    

    question_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=search_question,
    ).data[0].embedding

    retrieved = sorted(
        records,
        key=lambda record: cosine_similarity(
            question_embedding, record["embedding"]
        ),
        reverse=True,
    )[:top_k]

    context = "\n\n".join(
        f"[{number}] Source: {record['source']}, "
        f"chunk {record['chunk_number']}\n{record['text']}"
        for number, record in enumerate(retrieved, start=1)
    )

    response = client.responses.create(
        model=ANSWER_MODEL,
        instructions=(
            "Use earlier conversation to understand the current question. "
            "Earlier assistant answers are not verified evidence. "
            "Ground factual claims in the current retrieved passages. "
            "Use citation labels only from the current passages. "
            "If a reference remains ambiguous, ask for clarification. "
        ),
        input=(
            f"Earlier conversation:\n{history_text}\n\n"
            f"Retrieved passages:\n{context}\n\n"
            f"Current question:\n{question}"
        ),
        max_output_tokens=200,
        temperature=0.2,
    )

    return response.output_text, retrieved