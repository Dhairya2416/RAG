import json
import os
import re
import urllib.error
import urllib.request

import faiss
import numpy as np
from pypdf import PdfReader


def _load_local_env_file(env_path=".env"):
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env_file()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
LOW_COST_MODE = os.getenv("LOW_COST_MODE", "true").lower() in {"1", "true", "yes", "on"}
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1800"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "512"))

ENERGY_KEYWORDS = {
    "energy", "electricity", "grid", "power", "solar", "wind", "renewable",
    "generation", "transmission", "distribution", "voltage", "current",
    "fossil", "efficiency", "battery", "hydro", "nuclear", "utility",
}


def _ensure_configured():
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key missing. Set OPENAI_API_KEY.")


def _openai_post(path, payload):
    _ensure_configured()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{OPENAI_BASE_URL}{path}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI request failed: {e}") from e


def _embed_text(text, is_query=False):
    input_text = (text or "").strip()
    if not input_text:
        return None
    payload = {
        "model": OPENAI_EMBED_MODEL,
        "input": input_text,
    }
    if is_query:
        payload["input"] = f"Query: {input_text}"
    data = _openai_post("/embeddings", payload)
    return data["data"][0]["embedding"]


def _local_embed_text(text, dim=LOCAL_EMBED_DIM):
    tokens = _tokenize(text)
    vec = np.zeros(dim, dtype="float32")
    if not tokens:
        return vec.tolist()
    for tok in tokens:
        idx = hash(tok) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _keyword_retrieve(query, chunks, top_k=3):
    q_tokens = set(_tokenize(query))
    scored = []
    for i, chunk in enumerate(chunks):
        c_tokens = set(_tokenize(chunk))
        score = len(q_tokens & c_tokens)
        scored.append((score, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_indices = [i for _, i in scored[:max(1, min(top_k, len(chunks)))]]
    return [chunks[i] for i in best_indices]


def _keyword_energy_score(text):
    content = (text or "").lower()
    if not content:
        return 0.0
    hits = sum(1 for kw in ENERGY_KEYWORDS if kw in content)
    return hits / max(len(ENERGY_KEYWORDS), 1)


def _tokenize(text):
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _extractive_answer(context_chunks, question, max_sentences=3):
    context = " ".join(context_chunks)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    if not sentences:
        snippet = context.strip()
        return snippet[:400] if snippet else "Answer not found in the document."

    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return " ".join(sentences[:max_sentences])[:500]

    scored = []
    for sent in sentences:
        s_tokens = set(_tokenize(sent))
        overlap = len(q_tokens & s_tokens)
        if overlap > 0:
            scored.append((overlap, sent))

    if not scored:
        return " ".join(sentences[:max_sentences])[:500]

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [s for _, s in scored[:max_sentences]]
    return " ".join(best)[:700]


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def is_energy_document(sample_text, threshold=0.70):
    if not sample_text or not sample_text.strip():
        return False, 0.0

    keyword_score = _keyword_energy_score(sample_text)
    if keyword_score >= 0.12:
        return True, keyword_score

    try:
        doc_embedding = _embed_text(sample_text)
    except Exception:
        return keyword_score >= 0.08, keyword_score

    energy_reference = """
    Energy generation, electricity, power systems,
    renewable energy, solar power, wind energy,
    fossil fuels, energy efficiency, smart grids,
    electrical infrastructure, transmission, distribution,
    voltage, current, generation and consumption
    """
    try:
        energy_embedding = _embed_text(energy_reference)
    except Exception:
        return keyword_score >= 0.08, keyword_score

    doc_embedding = np.array(doc_embedding)
    energy_embedding = np.array(energy_embedding)
    denom = np.linalg.norm(doc_embedding) * np.linalg.norm(energy_embedding)
    if denom == 0:
        return False, 0.0

    similarity = np.dot(doc_embedding, energy_embedding) / denom
    return similarity >= threshold, similarity


def chunk_text(text, chunk_size=1400, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def generate_embeddings(chunks):
    valid_chunks = [c for c in chunks if c.strip()]
    if not valid_chunks:
        raise ValueError("No valid text chunks were found to embed.")

    embeddings = []
    use_local = False
    for chunk in valid_chunks:
        if use_local:
            embeddings.append(_local_embed_text(chunk))
            continue
        try:
            emb = _embed_text(chunk)
            if emb is not None:
                embeddings.append(emb)
        except Exception:
            # Quota or transient API failure: rebuild all embeddings locally
            use_local = True
            embeddings = [_local_embed_text(c) for c in valid_chunks]
            break

    if not embeddings:
        raise ValueError("No valid text chunks were found to embed.")

    return np.array(embeddings).astype("float32")


def build_vector_db(embeddings):
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(query, chunks, index, top_k=3):
    if not chunks:
        return []

    k = min(top_k, len(chunks))
    try:
        if index.d == LOCAL_EMBED_DIM:
            query_embedding = _local_embed_text(query)
        else:
            query_embedding = _embed_text(query, is_query=True)
        query_embedding = np.array([query_embedding]).astype("float32")
        _, indices = index.search(query_embedding, k)
        valid_indices = [i for i in indices[0] if 0 <= i < len(chunks)]
        if valid_indices:
            return [chunks[i] for i in valid_indices]
    except Exception:
        pass
    return _keyword_retrieve(query, chunks, top_k=k)


def generate_answer(context_chunks, question):
    if not context_chunks:
        return "Answer not found in the document."

    context = "\n\n".join(context_chunks)[:MAX_CONTEXT_CHARS]
    prompt = f"""
You are an energy-domain assistant.
Answer strictly from the context.
If not found, say: Answer not found in the document.

Context:
{context}

Question:
{question}
"""

    try:
        data = _openai_post(
            "/chat/completions",
            {
                "model": OPENAI_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": "You answer only from provided context."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 220 if LOW_COST_MODE else 400,
            },
        )
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return _extractive_answer(context_chunks, question)
        return _extractive_answer(context_chunks, question)
