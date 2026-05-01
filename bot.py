"""
DocBot — bot.py  (universal, no hardcoding)
============================================
Two-layer pipeline:
  Layer 1 — Excel fast-path  (any .xlsx/.xls, any content, any columns)
  Layer 2 — RAG + mistral    (any document type, any question)
"""

import os
import re
import shutil
import platform
from functools import lru_cache

import chromadb
import pandas as pd
from PIL import Image
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# ── Tesseract ────────────────────────────────────────────────────────────────
def _find_tesseract():
    if platform.system() == "Windows":
        c = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(c):
            return c
    return shutil.which("tesseract")

_tess = _find_tesseract()
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess


# ── Document loaders ─────────────────────────────────────────────────────────
def load_document(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
    elif ext in (".txt", ""):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in (".xls", ".xlsx"):
        docs = []
        for sheet, df in pd.read_excel(file_path, sheet_name=None).items():
            docs.append(Document(
                page_content=f"[Sheet: {sheet}]\n{df.to_string(index=False)}",
                metadata={"source": os.path.basename(file_path), "sheet": sheet}
            ))
        return docs
    elif ext in (".png", ".jpg", ".jpeg"):
        if not _tess:
            raise RuntimeError(
                "Tesseract not found. Install: https://github.com/tesseract-ocr/tesseract"
            )
        text = pytesseract.image_to_string(Image.open(file_path))
        return [Document(
            page_content=text,
            metadata={"source": os.path.basename(file_path)}
        )]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.basename(file_path)
    return docs


def load_documents_from_folder(folder_path: str) -> list:
    all_docs = []
    for fn in os.listdir(folder_path):
        fp = os.path.join(folder_path, fn)
        if os.path.isfile(fp):
            try:
                docs = load_document(fp)
                all_docs.extend(docs)
                print(f"✅ Loaded: {fn} ({len(docs)} page(s))")
            except Exception as e:
                print(f"⚠️  Skipped {fn}: {e}")
    return all_docs


# ── Embeddings ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _filter_chunks(chunks: list) -> list:
    """Remove only completely empty chunks."""
    return [c for c in chunks if c.page_content and c.page_content.strip()]


# ── ChromaDB helper ───────────────────────────────────────────────────────────
def _make_chroma_client(path: str):
    """
    Create a PersistentClient. If the folder is corrupted (old sqlite schema),
    rename it out of the way (Windows-safe — doesn't need file unlock)
    and create a fresh one. Falls back to EphemeralClient as last resort.
    """
    os.makedirs(path, exist_ok=True)
    for attempt in range(2):
        try:
            client = chromadb.PersistentClient(path=path)
            client.list_collections()  # smoke-test
            return client
        except Exception as e:
            if attempt == 0:
                print(f"⚠️  ChromaDB error ({e}), resetting folder...")
                bad = path + "_old"
                shutil.rmtree(bad, ignore_errors=True)
                try:
                    os.rename(path, bad)
                except Exception:
                    shutil.rmtree(path, ignore_errors=True)
                os.makedirs(path, exist_ok=True)
            else:
                print("⚠️  Using in-memory DB — documents re-indexed each session")
                return chromadb.EphemeralClient()


# ── RAG pipeline ─────────────────────────────────────────────────────────────
def process_documents(docs: list, folder_path: str, rag_settings: dict):
    """
    Index documents and return a ConversationalRetrievalChain.
    docs must already be filtered to exclude Excel files (done in app.py).
    """
    # Deduplicate docs by source+page to prevent double-indexing
    seen_keys = set()
    unique_docs = []
    for d in docs:
        key = (d.metadata.get("source", ""), d.metadata.get("page", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_docs.append(d)

    print(f"📄 Unique pages to index: {len(unique_docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = _filter_chunks(splitter.split_documents(unique_docs))
    print(f"🔪 Chunks after filtering: {len(chunks)}")

    if not chunks:
        # Don't crash — warn and return None so app stays usable for Excel
        print("⚠️  No readable text chunks found. PDF may be image-based.")
        return None

    embeddings        = get_embeddings()
    persist_directory = "chroma_db"
    collection_name   = "docbot"
    chroma_client     = _make_chroma_client(persist_directory)

    try:
        existing_names = [c.name for c in chroma_client.list_collections()]
    except Exception:
        existing_names = []

    if collection_name in existing_names:
        print("📦 Loading existing vector DB...")
        try:
            db = Chroma(client=chroma_client, collection_name=collection_name,
                        embedding_function=embeddings)
            existing_sources = {
                m["source"] for m in db.get()["metadatas"] if "source" in m
            }
            new_chunks = [c for c in chunks
                          if c.metadata.get("source") not in existing_sources]
            if new_chunks:
                db.add_documents(new_chunks)
                print(f"✅ {len(new_chunks)} new chunks added.")
            else:
                print("⚡ All documents already indexed.")
        except Exception as e:
            print(f"⚠️  Rebuilding DB from scratch ({e})...")
            existing_names = []

    if collection_name not in existing_names:
        print("🧠 Creating vector DB...")
        db = Chroma.from_documents(chunks, embeddings,
                                   client=chroma_client,
                                   collection_name=collection_name)
        print(f"✅ {len(chunks)} chunks indexed.")

    llm = Ollama(model="mistral", num_predict=600, temperature=0.0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    condense_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Rewrite the follow-up as a complete standalone question.
Keep ALL names, topics, keywords from the history.
Replace pronouns (he/she/they/it/him/his/her) with the actual name from history.
Output ONLY the rewritten question, nothing else.

History:
{chat_history}

Follow-up: {question}
Standalone question:""",
    )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a document assistant. Answer using ONLY the context below.

- "Who is X?" → summarise what the context says about X.
- "Who said [quote]?" → find the quote and return the name after it.
- "How many X by Y?" → count every Y in context, give number and list them.
- "Give me a quote about X" → find the most relevant quote mentioning X.
- "What did X say?" → list every quote/statement by X in the context.
- "Explain X" or "How does X work?" → explain using context details.
- If not in context → say: "The answer is not found in the uploaded documents."
- NEVER use outside knowledge.

Context:
{context}

Question: {question}
Answer:""",
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        ),
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
    )
    return qa


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — EXCEL FAST-PATH  (universal, zero domain hardcoding)
# ════════════════════════════════════════════════════════════════════════════

_STRUCTURAL_VERBS = {
    "list", "show", "find", "get", "fetch", "give", "display",
    "filter", "search", "lookup", "retrieve",
}
_DATA_REFERENCE_NOUNS = {
    "rows", "entries", "records", "items", "data", "values",
    "results", "cells", "columns",
}


def _is_excel_intent(question: str) -> bool:
    q = question.lower().strip()
    if re.search(
        r'\b(greater than|less than|more than|fewer than|equal to|'
        r'at least|at most|over|under|above|below)\s+[\d,]+', q):
        return True
    if re.search(r'\b(starting with|starts with|begins with)\s+[a-z]\b', q):
        return True
    if re.search(r'\b(spreadsheet|excel file|xlsx|in the sheet|from the sheet)\b', q):
        return True
    words = set(q.split())
    if not bool(words & _STRUCTURAL_VERBS):
        return False
    if bool(words & _DATA_REFERENCE_NOUNS):
        return True
    if re.search(r'\b(list|show|get|find|display)\s+all\b', q):
        return True
    return False


def smart_query_excel(folder_path: str, question: str) -> list:
    if not _is_excel_intent(question):
        return []

    q_lower  = question.lower()
    stop     = _STRUCTURAL_VERBS | _DATA_REFERENCE_NOUNS | {
        "the", "a", "an", "in", "of", "with", "from", "that",
        "where", "which", "what", "are", "is", "all", "any",
        "and", "or", "for", "to", "me", "give", "show",
    }
    keywords = [w for w in q_lower.split() if len(w) > 2 and w not in stop]
    results  = []

    for fn in os.listdir(folder_path):
        if not (fn.endswith(".xlsx") or fn.endswith(".xls")):
            continue
        try:
            sheets = pd.read_excel(
                os.path.join(folder_path, fn), sheet_name=None
            )
        except Exception as e:
            print(f"⚠️  Could not read {fn}: {e}")
            continue

        for sheet, df in sheets.items():
            df.columns = [str(c).lower().strip() for c in df.columns]
            if df.empty:
                continue
            header = f"{fn} › {sheet}"

            # Pattern 1: alphabetical filter
            if re.search(r'\b(starting with|starts with|begins with)\s+[a-z]\b', q_lower):
                m = re.search(
                    r'\b(?:starting with|starts with|begins with)\s+([a-z])\b', q_lower)
                if m:
                    letter   = m.group(1).upper()
                    best_col = _pick_best_text_column(df, keywords)
                    if best_col:
                        filt = df[df[best_col].astype(str).str.strip()
                                              .str.upper().str.startswith(letter)]
                        if not filt.empty:
                            vals = _dedup(filt[best_col].astype(str).str.strip().tolist())
                            results.append(f"{header}:\n" + "\n".join(vals[:30]))
                continue

            # Pattern 2: numeric comparison
            nm = re.search(
                r'\b(greater than|less than|more than|fewer than|equal to|'
                r'at least|at most|over|under|above|below)\s+([\d,]+\.?\d*)\b',
                q_lower)
            if nm:
                op_word = nm.group(1)
                val     = float(nm.group(2).replace(",", ""))
                op_map  = {
                    "greater than": lambda s, v=val: s > v,
                    "more than":    lambda s, v=val: s > v,
                    "over":         lambda s, v=val: s > v,
                    "above":        lambda s, v=val: s > v,
                    "at least":     lambda s, v=val: s >= v,
                    "less than":    lambda s, v=val: s < v,
                    "fewer than":   lambda s, v=val: s < v,
                    "under":        lambda s, v=val: s < v,
                    "below":        lambda s, v=val: s < v,
                    "at most":      lambda s, v=val: s <= v,
                    "equal to":     lambda s, v=val: s == v,
                }
                op_fn    = op_map.get(op_word, lambda s, v=val: s > v)
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                target   = next((c for c in num_cols if any(k in c for k in keywords)),
                                num_cols[0] if num_cols else None)
                if target:
                    filt     = df[op_fn(df[target])]
                    if not filt.empty:
                        disp = _pick_best_text_column(df, keywords) or df.columns[0]
                        vals = _dedup(filt[disp].astype(str).str.strip().tolist())
                        results.append(f"{header} [{target}]:\n" + "\n".join(vals[:30]))
                continue

            # Pattern 3: show/list all → whole best column
            if re.search(r'\b(list|show|get|find|display)\s+all\b', q_lower):
                best_col = _pick_best_text_column(df, keywords)
                if best_col:
                    vals = _dedup(
                        df[best_col].dropna().astype(str).str.strip().tolist())
                    vals = [v for v in vals if v and v.lower() != "nan"]
                    if vals:
                        results.append(f"{header}:\n" + "\n".join(vals[:50]))
                continue

            # Pattern 4: keyword match
            if keywords:
                matched = pd.DataFrame()
                col_hit = any(any(k in c for k in keywords)
                              for c in df.select_dtypes(include="object").columns)
                for col in df.select_dtypes(include="object").columns:
                    mask    = df[col].astype(str).str.lower().apply(
                        lambda x: any(k in x for k in keywords))
                    matched = pd.concat([matched, df[mask]]).drop_duplicates()

                if not matched.empty and (col_hit or len(matched) >= 2):
                    disp = _pick_best_text_column(matched, keywords) or matched.columns[0]
                    vals = _dedup(matched[disp].astype(str).str.strip().tolist())
                    vals = [v for v in vals if v and v.lower() != "nan"]
                    results.append(f"{header}:\n" + "\n".join(vals[:30]))

    return results


def _dedup(values: list) -> list:
    seen: set = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _pick_best_text_column(df: pd.DataFrame, keywords: list) -> str | None:
    obj_cols = list(df.select_dtypes(include="object").columns)
    if not obj_cols:
        return None
    for col in obj_cols:
        if any(k in col for k in keywords):
            return col
    for col in obj_cols:
        sample    = df[col].dropna().astype(str).head(15)
        if len(sample) == 0:
            continue
        name_like = sum(1 for v in sample
                        if 3 <= len(v) <= 80 and 1 <= len(v.split()) <= 10
                        and v[0].isupper())
        if name_like / len(sample) >= 0.5:
            return col
    return obj_cols[0]