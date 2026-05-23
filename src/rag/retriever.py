from src.graph.neo4j_client import Neo4jClient
from src.rag.embedder import get_embedding


class HybridRetriever:
    def __init__(self):
        self.client = Neo4jClient()

    def vector_search(self, query: str, top_k: int = 3) -> list[dict]:
        """Semantik benzerlik ile makale bul."""
        query_embedding = get_embedding(query)

        results = self.client.run_query("""
            CALL db.index.vector.queryNodes(
                'article_embeddings', $top_k, $embedding
            )
            YIELD node, score
            RETURN node.id AS id,
                   node.title AS title,
                   node.text AS text,
                   score
            ORDER BY score DESC
        """, {"top_k": top_k, "embedding": query_embedding})

        return results

    def graph_search(self, person_name: str) -> list[dict]:
        """Kişi adına göre ilgili olayları graph'tan getir."""
        results = self.client.run_query("""
            MATCH (p:Person)-[:INVOLVED_IN]->(e:Event)
            WHERE toLower(p.name) CONTAINS toLower($name)
            OPTIONAL MATCH (e)<-[:CONTAINS_EVENT]-(a:Article)
            OPTIONAL MATCH (other:Person)-[:INVOLVED_IN]->(e)
            WHERE other.name <> p.name
            RETURN p.name AS person,
                   e.description AS event,
                   e.relation AS relation,
                   a.title AS article,
                   collect(DISTINCT other.name) AS other_persons
        """, {"name": person_name})

        return results

    def multi_hop_search(self, person_name: str) -> list[dict]:
        """2 adımlı graph traversal — kişinin bağlantılı olduğu diğer kişiler."""
        results = self.client.run_query("""
            MATCH (p:Person)-[:INVOLVED_IN]->(e:Event)<-[:INVOLVED_IN]-(other:Person)
            WHERE toLower(p.name) CONTAINS toLower($name)
            AND other.name <> p.name
            RETURN p.name AS source_person,
                   other.name AS connected_person,
                   e.description AS via_event,
                   count(e) AS shared_events
            ORDER BY shared_events DESC
        """, {"name": person_name})

        return results

    def hybrid_search(self, query: str, person_hint: str = "") -> dict:
        """Vector + Graph aramasını birleştir."""
        print(f"\n🔍 Sorgu: {query}")

        # 1. Vector search
        print("  → Vector search çalışıyor...")
        vector_results = self.vector_search(query, top_k=3)

        # 2. Graph search (kişi ipucu varsa)
        graph_results = []
        multi_hop = []

        if person_hint:
            print(f"  → Graph search: '{person_hint}'...")
            graph_results = self.graph_search(person_hint)
            multi_hop = self.multi_hop_search(person_hint)

        return {
            "vector_results": vector_results,
            "graph_results": graph_results,
            "multi_hop": multi_hop
        }

    def close(self):
        self.client.close()


if __name__ == "__main__":
    from src.graph.graph_writer import GraphWriter
    from src.rag.embedder import embed_article
    from src.ingestion.ner_extractor import extract_all

    # Önce article'a embedding ekle
    print("=== Embedding ekleniyor ===")
    writer = GraphWriter()
    embed_article(
        "article_001",
        "Erdoğan Biden ile NATO zirvesinde bir araya geldi. Ukrayna krizi ele alındı.",
        writer.client
    )
    writer.close()

    # Hybrid search test
    print("\n=== Hybrid Search Test ===")
    retriever = HybridRetriever()

    result = retriever.hybrid_search(
        query="NATO zirvesinde kimler görüştü?",
        person_hint="Erdoğan"
    )

    print("\n--- Vector Sonuçları ---")
    for r in result["vector_results"]:
        print(f"  [{r.get('score', 0):.3f}] {r.get('title', '')}")

    print("\n--- Graph Sonuçları (Erdoğan'ın olayları) ---")
    for r in result["graph_results"]:
        print(f"  • {r['event']}")
        print(f"    Diğer kişiler: {r['other_persons']}")

    print("\n--- Multi-hop (Erdoğan'ın bağlantıları) ---")
    for r in result["multi_hop"]:
        print(f"  {r['source_person']} ←→ {r['connected_person']} ({r['shared_events']} ortak olay)")

    retriever.close()