from src.graph.neo4j_client import Neo4jClient
from datetime import datetime
import hashlib


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


class GraphWriter:
    def __init__(self):
        self.client = Neo4jClient()

    def write_article(self, article_id: str, title: str, text: str, url: str = ""):
        self.client.run_query("""
            MERGE (a:Article {id: $id})
            SET a.title = $title,
                a.text = $text,
                a.url = $url,
                a.created_at = $created_at
        """, {
            "id": article_id,
            "title": title,
            "text": text,
            "url": url,
            "created_at": datetime.now().isoformat()
        })
        print(f"  ✓ Article kaydedildi: {title[:50]}")

    def write_entities(self, article_id: str, entities: dict):
        # Kişiler
        for person in entities.get("persons", []):
            self.client.run_query("""
                MERGE (p:Person {name: $name})
                WITH p
                MATCH (a:Article {id: $article_id})
                MERGE (a)-[:MENTIONS_PERSON]->(p)
            """, {"name": person, "article_id": article_id})

        # Organizasyonlar
        for org in entities.get("organizations", []):
            self.client.run_query("""
                MERGE (o:Organization {name: $name})
                WITH o
                MATCH (a:Article {id: $article_id})
                MERGE (a)-[:MENTIONS_ORG]->(o)
            """, {"name": org, "article_id": article_id})

        # Lokasyonlar
        for loc in entities.get("locations", []):
            self.client.run_query("""
                MERGE (l:Location {name: $name})
                WITH l
                MATCH (a:Article {id: $article_id})
                MERGE (a)-[:MENTIONS_LOCATION]->(l)
            """, {"name": loc, "article_id": article_id})

        print(f"  ✓ Entities yazıldı: "
              f"{len(entities.get('persons', []))} kişi, "
              f"{len(entities.get('organizations', []))} org, "
              f"{len(entities.get('locations', []))} lokasyon")

    def write_events(self, article_id: str, events: list):
        for event in events:
            event_id = generate_id(event["event"])

            # Event node oluştur
            self.client.run_query("""
                MERGE (e:Event {id: $id})
                SET e.description = $description,
                    e.location = $location,
                    e.date = $date,
                    e.relation = $relation
                WITH e
                MATCH (a:Article {id: $article_id})
                MERGE (a)-[:CONTAINS_EVENT]->(e)
            """, {
                "id": event_id,
                "description": event["event"],
                "location": event.get("location", ""),
                "date": event.get("date", ""),
                "relation": event.get("relation", ""),
                "article_id": article_id
            })

            # Kişileri event'e bağla
            for person in event.get("persons", []):
                self.client.run_query("""
                    MERGE (p:Person {name: $name})
                    WITH p
                    MATCH (e:Event {id: $event_id})
                    MERGE (p)-[:INVOLVED_IN {relation: $relation}]->(e)
                """, {
                    "name": person,
                    "event_id": event_id,
                    "relation": event.get("relation", "")
                })

            # Organizasyonları event'e bağla
            for org in event.get("organizations", []):
                self.client.run_query("""
                    MERGE (o:Organization {name: $name})
                    WITH o
                    MATCH (e:Event {id: $event_id})
                    MERGE (o)-[:INVOLVED_IN]->(e)
                """, {"name": org, "event_id": event_id})

        print(f"  ✓ {len(events)} event yazıldı")

    def close(self):
        self.client.close()


if __name__ == "__main__":
    from src.ingestion.ner_extractor import extract_all

    test_article = {
        "id": "article_001",
        "title": "Erdoğan Biden ile NATO Zirvesinde Bir Araya Geldi",
        "text": """
            Cumhurbaşkanı Erdoğan, Ankara'da düzenlenen NATO zirvesinde 
            ABD Başkanı Biden ile bir araya geldi. Toplantıda Ukrayna krizi 
            ve savunma harcamaları ele alındı. Dışişleri Bakanı Fidan da 
            görüşmelere katılırken, Türkiye'nin F-16 talebi de gündeme geldi.
        """,
        "url": "https://example.com/haber/001"
    }

    print("=== Graph Writer Test ===\n")
    writer = GraphWriter()

    print("1. Article yazılıyor...")
    writer.write_article(
        test_article["id"],
        test_article["title"],
        test_article["text"],
        test_article["url"]
    )

    print("2. Entities çıkarılıyor...")
    result = extract_all(test_article["text"])

    print("3. Entities Neo4j'e yazılıyor...")
    writer.write_entities(test_article["id"], result["entities"])

    print("4. Events Neo4j'e yazılıyor...")
    writer.write_events(test_article["id"], result["events"])

    writer.close()

    print("\n=== Tamamlandı! Neo4j Browser'dan kontrol et ===")
    print("http://localhost:7474")
    print("Cypher: MATCH (n) RETURN n LIMIT 50")