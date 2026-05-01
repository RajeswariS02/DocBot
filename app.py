import os
import shutil
import time

import streamlit as st

from bot import load_documents_from_folder, process_documents, smart_query_excel

st.set_page_config(page_title="DocBot", page_icon="📚", layout="wide")
st.title("📚 DocBot — Chat With Your Documents")
st.caption("Local RAG · No API keys · Powered by Ollama + ChromaDB + HuggingFace")

# ── Constants & helpers (defined before sidebar) ─────────────────────────────
DOCUMENT_FOLDER  = "documents"
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)


def _safe_delete(path: str, retries: int = 5) -> None:
    for _ in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(1)


def _preview_document(file_name: str) -> None:
    file_path = os.path.join(DOCUMENT_FOLDER, file_name)
    ext = os.path.splitext(file_name)[1].lower()
    if ext in (".xlsx", ".xls"):
        import pandas as pd
        xl = pd.read_excel(file_path, sheet_name=None)
        for sheet, df in xl.items():
            st.markdown(f"**Sheet: {sheet}**")
            st.dataframe(df, use_container_width=True)
    elif ext in (".jpg", ".jpeg", ".png"):
        st.image(file_path)
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            st.text(f.read(3000))


def _format_excel_answer(results: list) -> str:
    lines = []
    for block in results:
        parts = block.split(":\n", 1)
        if len(parts) == 2:
            header, names = parts
            name_list = [n.strip() for n in names.strip().split("\n") if n.strip()]
            lines.append(f"**{header}**")
            for name in name_list:
                lines.append(f"- {name}")
        else:
            lines.append(block)
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ RAG Settings")
    chunk_size    = st.slider("Chunk size (chars)",    200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)",   0,  300, 100, step=25)
    top_k         = st.slider("Top-k chunks",            1,   10,   3)
    st.divider()
    st.markdown("**Model:** `mistral` via Ollama")
    st.markdown("**Embeddings:** `all-MiniLM-L6-v2`")
    st.markdown("**Vector DB:** ChromaDB (local)")
    if st.button("🗑️ Clear vector DB & chat"):
        _safe_delete("chroma_db")
        _safe_delete(DOCUMENT_FOLDER)
        st.session_state.clear()
        st.rerun()

# ── Settings change detection ─────────────────────────────────────────────────
current_settings = {
    "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "top_k": top_k
}
if st.session_state.get("rag_settings") != current_settings:
    st.session_state["rag_settings"] = current_settings
    st.session_state["qa_chain"]     = None

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT, XLSX, PNG, JPG)",
    type=["pdf", "docx", "txt", "xlsx", "xls", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["qa_chain"] = None
    _safe_delete(DOCUMENT_FOLDER)
    _safe_delete("chroma_db")
    os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
    for uf in uploaded_files:
        with open(os.path.join(DOCUMENT_FOLDER, uf.name), "wb") as f:
            f.write(uf.getbuffer())
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully.")

# ── Document preview ──────────────────────────────────────────────────────────
if os.path.exists(DOCUMENT_FOLDER):
    files = [f for f in os.listdir(DOCUMENT_FOLDER)
             if os.path.isfile(os.path.join(DOCUMENT_FOLDER, f))]
    if files:
        with st.expander(f"📂 Uploaded files ({len(files)})"):
            selected = st.selectbox("Preview a file", ["— select —"] + files)
            if selected != "— select —":
                _preview_document(selected)

# ── Build QA chain (skips Excel files from RAG) ───────────────────────────────
if st.session_state.get("qa_chain") is None:
    if os.path.exists(DOCUMENT_FOLDER):
        all_files = [f for f in os.listdir(DOCUMENT_FOLDER)
                     if os.path.isfile(os.path.join(DOCUMENT_FOLDER, f))]
        non_excel  = [f for f in all_files
                      if os.path.splitext(f)[1].lower() not in EXCEL_EXTENSIONS]
        if non_excel:
            docs = load_documents_from_folder(DOCUMENT_FOLDER)
            docs = [d for d in docs
                    if os.path.splitext(d.metadata.get("source", ""))[1].lower()
                    not in EXCEL_EXTENSIONS]
            if docs:
                with st.spinner("⚡ Building vector index…"):
                    result = process_documents(
                        docs, DOCUMENT_FOLDER, st.session_state["rag_settings"]
                    )
                if result is not None:
                    st.session_state["qa_chain"] = result
                    st.success("✅ Documents indexed — ready to chat!")
                else:
                    st.warning(
                        "⚠️ Could not extract text from your PDF. "                        "It may be a scanned/image PDF. "                        "Excel files can still be queried directly."
                    )
            else:
                st.info("No indexable documents found.")
        else:
            st.info("Upload documents above to get started.")

# ── Chat state ────────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# ── Chat input ────────────────────────────────────────────────────────────────
user_question = st.chat_input("Ask something about your documents…")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):

            # Build history pairs for conversational context
            history_pairs = []
            msgs = st.session_state.chat_history[:-1]
            for i in range(0, len(msgs) - 1, 2):
                if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                    history_pairs.append(
                        (msgs[i]["content"], msgs[i+1]["content"])
                    )

            # Layer 1: Excel fast-path (structured/tabular queries)
            excel_results = smart_query_excel(DOCUMENT_FOLDER, user_question)

            if excel_results:
                answer  = _format_excel_answer(excel_results)
                sources = ["Excel (direct query)"]

            elif st.session_state.get("qa_chain") is None:
                answer  = "Please upload some documents first."
                sources = []

            else:
                # Layer 2: RAG + LLM (handles all document types and question types)
                response = st.session_state["qa_chain"].invoke({
                    "question":     user_question,
                    "chat_history": history_pairs,
                })
                answer = response["answer"]

                seen = {}
                for doc in response.get("source_documents", []):
                    fname = doc.metadata.get("source", "Unknown")
                    if os.path.splitext(fname)[1].lower() in EXCEL_EXTENSIONS:
                        continue
                    page  = doc.metadata.get("page")
                    sheet = doc.metadata.get("sheet")
                    seen.setdefault(fname, set())
                    if page  is not None: seen[fname].add(f"p.{page + 1}")
                    if sheet is not None: seen[fname].add(f"sheet:{sheet}")

                sources = [
                    fname + (f" ({', '.join(sorted(tags))})" if tags else "")
                    for fname, tags in seen.items()
                ]

        st.markdown(answer)
        if sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": sources,
    })