import json
import os
import hashlib
from pathlib import Path
from src.ingestion.ner_extractor import extract_all
from src.graph.graph_writer import GraphWriter
from src.rag.embedder import embed_article

def generate_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def ingest_article(article: dict, writer: GraphWriter) -> bool:
    try:
        article_id = article.get("id") or generate_id(article["title"])
        print(f"\n📰 {article['title'][:60]}")

        # 1. Article kaydet
        writer.write_article(
            article_id,
            article["title"],
            article["text"],
            article.get("url", "")
        )

        # 2. NER
        result = extract_all(article["text"])

        # 3. Entities yaz
        writer.write_entities(article_id, result["entities"])

        # 4. Events yaz
        writer.write_events(article_id, result["events"])

        # 5. Embedding ekle
        embed_article(article_id, article["title"] + " " + article["text"], writer.client)

        return True

    except Exception as e:
        print(f"  ✗ Hata: {e}")
        return False


def ingest_from_json(filepath: str):
    """JSON dosyasından toplu haber yükle."""
    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"=== {len(articles)} haber yükleniyor ===")
    writer = GraphWriter()

    success, fail = 0, 0
    for article in articles:
        if ingest_article(article, writer):
            success += 1
        else:
            fail += 1

    writer.close()
    print(f"\n✓ Tamamlandı: {success} başarılı, {fail} başarısız")


if __name__ == "__main__":
    ingest_from_json("data/sample_news/news.json")