import spacy
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

nlp = spacy.load("xx_ent_wiki_sm")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL  = os.getenv("LLM_MODEL", "llama3.2")


def extract_with_spacy(text: str) -> dict:
    """Hızlı ilk geçiş — spaCy ile temel entity'leri çıkar."""
    doc = nlp(text)
    entities = {"persons": [], "organizations": [], "locations": [], "misc": []}

    for ent in doc.ents:
        if ent.label_ in ("PER", "PERSON"):
            entities["persons"].append(ent.text.strip())
        elif ent.label_ in ("ORG",):
            entities["organizations"].append(ent.text.strip())
        elif ent.label_ in ("LOC", "GPE"):
            entities["locations"].append(ent.text.strip())
        else:
            entities["misc"].append(ent.text.strip())

    # Tekrarları kaldır
    for k in entities:
        entities[k] = list(set(entities[k]))

    return entities


def extract_events_with_llm(text: str, entities: dict) -> list[dict]:
    """Ollama ile olayları ve ilişkileri çıkar."""
    prompt = f"""Analyze this news text and extract events in JSON format.

Text:
{text[:1500]}

Detected entities:
- Persons: {', '.join(entities['persons'])}
- Organizations: {', '.join(entities['organizations'])}
- Locations: {', '.join(entities['locations'])}

Return ONLY a JSON array (no wrapper object, no explanation):
[
  {{
    "event": "short event summary",
    "persons": ["involved persons"],
    "organizations": ["related organizations"],
    "location": "event location or null",
    "date": "date if mentioned or null",
    "relation": "person-event relation (e.g. announced, attended, accused)"
  }}
]"""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 2048,   # truncate sorunu çözülür  # 1024 → 2048
                "temperature": 0.1     # daha tutarlı JSON çıktısı
            }
        },
        timeout=120.0
    )

    raw = response.json()["response"].strip()

    try:
        # ```json bloğunu temizle
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        # Wrapper dict handle et
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Kesik JSON'ı kurtar — son geçerli objeyi bul
            # Tamamlanmış { } bloklarını tek tek çek
            events = []
            depth = 0
            start = None
            for i, ch in enumerate(raw):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        chunk = raw[start:i+1]
                        try:
                            events.append(json.loads(chunk))
                        except json.JSONDecodeError:
                            pass
            if events:
                print(f"  ⚠ Kesik JSON kurtarıldı: {len(events)} event")
                return events
            print(f"LLM JSON parse hatası, ham çıktı:\n{raw}")
            return []

        if isinstance(parsed, dict):
            for key in ("events", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return []

        return parsed if isinstance(parsed, list) else []

    except Exception as e:
        print(f"Parse hatası: {e}\nHam çıktı:\n{raw}")
        return []

def extract_all(text: str) -> dict:
    """Ana fonksiyon — spaCy + LLM birleşik çıkarma."""
    print("  → spaCy ile entity çıkarılıyor...")
    entities = extract_with_spacy(text)

    print("  → Ollama ile olaylar çıkarılıyor...")
    events = extract_events_with_llm(text, entities)

    return {
        "entities": entities,
        "events": events
    }


if __name__ == "__main__":
    test_text = """
    Cumhurbaşkanı Erdoğan, Ankara'da düzenlenen NATO zirvesinde 
    ABD Başkanı Biden ile bir araya geldi. Toplantıda Ukrayna krizi 
    ve savunma harcamaları ele alındı. Dışişleri Bakanı Fidan da 
    görüşmelere katılırken, Türkiye'nin F-16 talebi de gündeme geldi.
    """

    result = extract_all(test_text)
    print("\n=== SONUÇ ===")
    print("Entities:", result["entities"])
    print("Events:", json.dumps(result["events"], ensure_ascii=False, indent=2))