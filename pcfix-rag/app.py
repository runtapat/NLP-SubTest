"""PC-Fix RAG: แชตบอตตอบปัญหา Hardware คอมพิวเตอร์ ด้วยเทคนิค RAG
Pipeline: Load -> Clean -> Chunk -> Embed -> FAISS -> Prompt -> Groq LLM -> Chat UI
"""
import glob
import os
import re

import faiss
import numpy as np
import streamlit as st
from pythainlp.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # รองรับไทย+อังกฤษ
LLM_MODEL = "llama-3.3-70b-versatile"
CHUNK_SIZE = 500      # ตัวอักษรต่อ chunk
CHUNK_OVERLAP = 80    # ตัวอักษรที่ซ้อนทับ
TOP_K = 4
MIN_SCORE = 0.25      # ต่ำกว่านี้ถือว่าเอกสารไม่เกี่ยวข้อง (ที่เหลือให้ Prompt ตัดสินว่า 'ไม่พบข้อมูล')

SYSTEM_PROMPT = """คุณคือ "PC-Fix" ผู้ช่วยแก้ปัญหา Hardware คอมพิวเตอร์
กติกา:
1. ตอบจาก [CONTEXT] ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดาเอง
2. ถ้า CONTEXT ไม่มีข้อมูลที่ตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร" เท่านั้น
3. ตอบเป็นภาษาเดียวกับคำถาม (ไทยหรืออังกฤษ) กระชับ เป็นขั้นตอนถ้าเป็นวิธีแก้
4. ท้ายคำตอบให้ระบุแหล่งที่มาเป็น [ชื่อไฟล์] ของ CONTEXT ที่ใช้จริง
5. ถ้าเกี่ยวกับความปลอดภัย (ไฟฟ้า แบตบวม) ให้เตือนผู้ใช้ตามที่เอกสารระบุ"""


# ---------- 1) Document Loading & Chunking ----------
def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_units(paragraph: str):
    """แบ่งย่อหน้าเป็นประโยค (ไทย/อังกฤษ) ด้วย PyThaiNLP"""
    return [s.strip() for s in sent_tokenize(paragraph, engine="whitespace+newline") if s.strip()]


def chunk_document(name: str, text: str):
    chunks, buf = [], ""
    for para in clean_text(text).split("\n"):
        for unit in split_units(para) if len(para) > CHUNK_SIZE else [para]:
            if not unit.strip():
                continue
            if len(buf) + len(unit) + 1 > CHUNK_SIZE and buf:
                chunks.append(buf.strip())
                tail = buf[-CHUNK_OVERLAP:]
                buf = (tail.split(" ", 1)[-1] if " " in tail else tail) + "\n"  # overlap (ตัดที่ขอบคำ)
            buf += unit + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return [{"source": name, "text": c} for c in chunks]


def load_chunks():
    out = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            out += chunk_document(os.path.basename(path), f.read())
    return out


# ---------- 2) Embedding & Vector Search ----------
@st.cache_resource(show_spinner="กำลังโหลดโมเดลและสร้างดัชนีเวกเตอร์...")
def build_index():
    chunks = load_chunks()
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine (normalized)
    index.add(emb)
    return model, index, chunks


def retrieve(query: str, k: int = TOP_K):
    model, index, chunks = build_index()
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(q, k)
    return [{**chunks[i], "score": float(s)} for s, i in zip(scores[0], ids[0]) if i >= 0]


# ---------- 3) Prompt Engineering + 4) LLM ----------
def build_messages(question, hits, history):
    context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += history[-6:]  # จำบทสนทนาล่าสุด 3 รอบ
    msgs.append({"role": "user", "content": f"[CONTEXT]\n{context}\n\n[QUESTION]\n{question}"})
    return msgs


PREFERRED_MODELS = [LLM_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                    "llama-3.1-8b-instant", "qwen/qwen3-32b"]


def pick_model(client):
    """เลือกโมเดลแรกที่บัญชี Groq ของผู้ใช้เรียกใช้ได้จริง (กัน model_not_found)"""
    if "llm_model" in st.session_state:
        return st.session_state["llm_model"]
    try:
        available = {m.id for m in client.models.list().data}
    except Exception:
        available = set()
    skip = ("whisper", "tts", "guard", "orpheus", "playai", "embed")
    chosen = next((m for m in PREFERRED_MODELS if m in available), None) or next(
        (m for m in sorted(available) if not any(k in m for k in skip)), LLM_MODEL)
    st.session_state["llm_model"] = chosen
    return chosen


def ask_llm(messages):
    from groq import Groq
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])  # เก็บใน Streamlit Secrets เท่านั้น
    r = client.chat.completions.create(model=pick_model(client), messages=messages, temperature=0.1)
    return r.choices[0].message.content


def answer(question, history):
    hits = [h for h in retrieve(question) if h["score"] >= MIN_SCORE]
    if not hits:
        return "ไม่พบข้อมูลในเอกสาร", []
    try:
        has_key = bool(st.secrets["GROQ_API_KEY"])
    except Exception:
        has_key = False
    if not has_key:  # ยังไม่ตั้งค่า Secrets: แสดงเฉพาะผลการค้นหา
        return "⚠️ ยังไม่ได้ตั้งค่า `GROQ_API_KEY` ใน Streamlit Secrets จึงแสดงเฉพาะผลการค้นหาเอกสาร", hits
    return ask_llm(build_messages(question, hits, history)), hits


# ---------- 5) Chatbot Interface ----------
st.set_page_config(page_title="PC-Fix RAG", page_icon="🛠️")
st.title("🛠️ PC-Fix: ผู้ช่วยแก้ปัญหา Hardware คอมพิวเตอร์")
st.caption("ตอบจากคลังเอกสาร 12 ไฟล์ (ไทย/อังกฤษ) ด้วยเทคนิค RAG · ข้อมูลในเอกสารเป็นข้อมูลจำลองเพื่อการศึกษา")

with st.sidebar:
    st.header("ตัวอย่างคำถาม")
    examples = ["เปิดเครื่องไม่ติดต้องเช็กอะไรบ้าง",
                "CPU อุณหภูมิเท่าไหร่ถึงอันตราย",
                "แบตเตอรี่โน้ตบุ๊กบวมต้องทำอย่างไร",
                "What does a WHEA_UNCORRECTABLE_ERROR mean?"]
    for e in examples:
        if st.button(e, use_container_width=True):
            st.session_state["pending"] = e
    if st.button("🗑️ ล้างแชต", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown(f"**Embedding:** `{EMBED_MODEL}`  \n**LLM:** `{st.session_state.get('llm_model', LLM_MODEL)}` (Groq)  \n**Top-K:** {TOP_K}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            with st.expander("📚 เอกสารอ้างอิง"):
                for h in m["hits"]:
                    st.markdown(f"**{h['source']}** (similarity {h['score']:.2f})")
                    st.text(h["text"])

prompt = st.chat_input("พิมพ์คำถามเกี่ยวกับปัญหา Hardware...") or st.session_state.pop("pending", None)
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาและเรียบเรียงคำตอบ..."):
            try:
                reply, hits = answer(prompt, history)
            except Exception as ex:
                reply, hits = f"เกิดข้อผิดพลาดในการเรียก LLM: {ex}", []
        st.markdown(reply)
        if hits:
            with st.expander("📚 เอกสารอ้างอิง", expanded=True):
                for h in hits:
                    st.markdown(f"**{h['source']}** (similarity {h['score']:.2f})")
                    st.text(h["text"])
    st.session_state.messages.append({"role": "assistant", "content": reply, "hits": hits})
