import httpx
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list[float]:
    """Tek metin için embedding üret."""
    response = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30.0
    )
    return response.json()["embedding"]


def embed_article(article_id: str, text: str, client) -> None:
    """Article node'una embedding ekle."""
    embedding = get_embedding(text)

    client.run_query("""
        MATCH (a:Article {id: $id})
        SET a.embedding = $embedding
    """, {"id": article_id, "embedding": embedding})

    print(f"  ✓ Embedding eklendi: {len(embedding)} boyut")


if __name__ == "__main__":
    test = "Erdoğan Biden ile NATO zirvesinde bir araya geldi."
    emb = get_embedding(test)
    print(f"Embedding boyutu: {len(emb)}")
    print(f"İlk 5 değer: {emb[:5]}")