from src.graph.neo4j_client import Neo4jClient

SCHEMA_QUERIES = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
    """
    CREATE VECTOR INDEX article_embeddings IF NOT EXISTS
    FOR (a:Article) ON (a.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
]

def setup_schema():
    client = Neo4jClient()
    for query in SCHEMA_QUERIES:
        client.run_query(query)
        print(f"✓ {query.strip()[:60]}...")
    client.close()
    print("\nSchema kurulumu tamamlandı.")

if __name__ == "__main__":
    setup_schema()