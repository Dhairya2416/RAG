import os
import hashlib
import streamlit as st

from rag_utils import (
    extract_text_from_pdf,
    is_energy_document,
    chunk_text,
    generate_embeddings,
    build_vector_db,
    retrieve_chunks,
    generate_answer,
    LOW_COST_MODE,
)


st.set_page_config(page_title="Energy RAG Chatbot", layout="wide")

st.title("Energy Document Chat (RAG)")
st.write("Upload **energy-related PDFs only**. Non-energy files are automatically rejected.")
if LOW_COST_MODE:
    st.caption("Low-cost mode is enabled: reduced context and cached answers to save quota.")

if not os.getenv("OPENAI_API_KEY"):
    st.warning(
        "OpenAI API key not found. Set OPENAI_API_KEY before using the app."
    )

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("data", exist_ok=True)
    pdf_path = f"data/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully.")

    try:
        with st.spinner("Checking if document is energy-related..."):
            full_text = extract_text_from_pdf(pdf_path)
            sample_text = full_text[:2000]
            is_energy, score = is_energy_document(sample_text)
    except Exception as e:
        st.error(f"Failed while checking the document: {e}")
        st.stop()

    if not full_text or not full_text.strip():
        st.error("No readable text found in this PDF.")
        st.stop()

    if not is_energy:
        st.error(f"Not an energy document (similarity score: {score:.2f})")
        st.stop()

    st.success(f"Energy document confirmed (similarity score: {score:.2f})")
    doc_id = hashlib.md5(full_text.encode("utf-8")).hexdigest()
    if st.session_state.active_doc_id != doc_id:
        st.session_state.answer_cache = {}
    st.session_state.active_doc_id = doc_id

    try:
        with st.spinner("Building RAG pipeline..."):
            chunks = chunk_text(full_text)
            embeddings = generate_embeddings(chunks)
            st.session_state.vector_db = build_vector_db(embeddings)
            st.session_state.chunks = chunks
    except Exception as e:
        st.error(f"Failed to build the RAG pipeline: {e}")
        st.stop()

    st.success("RAG pipeline ready.")

st.markdown("### Ask a question from the document")

question = st.text_input(
    "Enter your question (energy-related):",
    placeholder="e.g. What is electricity?",
)

if question:
    if st.session_state.vector_db is None:
        st.warning("Please upload an energy-related PDF first.")
    else:
        try:
            with st.spinner("Generating answer..."):
                normalized_question = question.strip().lower()
                cache_key = f"{st.session_state.active_doc_id}:{normalized_question}"
                cached = st.session_state.answer_cache.get(cache_key)

                if cached:
                    retrieved_chunks = cached["retrieved_chunks"]
                    answer = cached["answer"]
                else:
                    retrieved_chunks = retrieve_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.vector_db,
                        top_k=2 if LOW_COST_MODE else 3,
                    )
                    answer = generate_answer(retrieved_chunks, question)
                    st.session_state.answer_cache[cache_key] = {
                        "retrieved_chunks": retrieved_chunks,
                        "answer": answer,
                    }
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
            st.stop()

        st.markdown("### Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for chunk in retrieved_chunks:
                st.write(chunk[:400] + "...")
