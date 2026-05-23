import httpx
import os
from dotenv import load_dotenv
from src.rag.retriever import HybridRetriever

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL  = os.getenv("LLM_MODEL", "llama3.2")


def build_context(retrieval_result: dict) -> str:
    """Vector + graph sonuçlarını LLM için tek context'e dönüştür."""
    parts = []

    # Vector sonuçları
    if retrieval_result["vector_results"]:
        parts.append("=== İLGİLİ HABERLER ===")
        for r in retrieval_result["vector_results"]:
            parts.append(f"Başlık: {r.get('title', '')}")
            parts.append(f"İçerik: {r.get('text', '').strip()}")
            parts.append("")

    # Graph sonuçları
    if retrieval_result["graph_results"]:
        parts.append("=== OLAY GRAFİĞİ ===")
        for r in retrieval_result["graph_results"]:
            other = ", ".join(r["other_persons"]) if r["other_persons"] else "—"
            parts.append(f"• Olay: {r['event']}")
            parts.append(f"  Diğer katılımcılar: {other}")
            parts.append("")

    # Multi-hop bağlantılar
    if retrieval_result["multi_hop"]:
        parts.append("=== KİŞİ BAĞLANTILARI ===")
        seen = set()
        for r in retrieval_result["multi_hop"]:
            key = tuple(sorted([r["source_person"], r["connected_person"]]))
            if key not in seen:
                seen.add(key)
                parts.append(
                    f"• {r['source_person']} ↔ {r['connected_person']} "
                    f"({r['shared_events']} ortak olay: {r['via_event'][:60]}...)"
                )

    return "\n".join(parts)


def ask(question: str, person_hint: str = "") -> str:
    """Soruyu hybrid retrieval + LLM ile yanıtla."""
    retriever = HybridRetriever()

    # Retrieval
    retrieval_result = retriever.hybrid_search(question, person_hint)
    retriever.close()

    # Context oluştur
    context = build_context(retrieval_result)

    if not context.strip():
        return "Bu konuda veritabanında yeterli bilgi bulunamadı."

#   # LLM prompt
#         prompt = f"""Sen bir Türkçe haber analiz asistanısın.
#     Aşağıdaki bilgileri kullanarak soruyu yanıtla.

#     KURALLAR:
#     - Sadece Türkçe yaz, kesinlikle başka dil karıştırma
#     - Bilgileri özetle, birebir kopyalama
#     - Emin olmadığın şeyi yazma

#     {context}

#     SORU: {question}

#     TÜRKÇE YANIT:""" 
   
    prompt = f"""You are a news analysis assistant. Answer the question using ONLY the information provided below.
Rules:
- Respond ONLY in English
- Do not use any other language
- Do not speculate beyond the given information
- Be concise and factual

{context}

QUESTION: {question}

ENGLISH ANSWER:"""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 512,
                "temperature": 0.2
            }
        },
        timeout=120.0
    )

    return response.json()["response"].strip()


if __name__ == "__main__":
    print("=" * 50)
    print("NewsGraphRAG — Soru Cevap Sistemi")
    print("=" * 50)

    sorular = [
        ("NATO zirvesinde kimler bir araya geldi?", "Erdoğan"),
        ("Fidan hangi olaylarda yer aldı?", "Fidan"),
        ("Toplantıda hangi konular ele alındı?", ""),
    ]

    for soru, kisi in sorular:
        print(f"\n❓ {soru}")
        cevap = ask(soru, kisi)
        print(f"💬 {cevap}")
        print("-" * 50)