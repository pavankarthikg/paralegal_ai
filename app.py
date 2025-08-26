import os, io, json, datetime, tempfile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
from ics import Calendar, Event

# ---- LLM (OpenAI) ----
from openai import OpenAI

# ---------------------- Config ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

st.set_page_config(page_title="Paralegal AI (POC)", page_icon="⚖️", layout="wide")
st.title("⚖️ Paralegal Assistant (POC)")
st.caption("Informational tool; not legal advice. Runs locally. Your files stay on your machine.")

# ---------------------- Models (cache) ----------------------
@st.cache_resource
def get_embedder():
    # multilingual model works well for Telugu/English
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def get_whisper_model():
    # small or medium for better Telugu; change to "small" on low RAM
    return WhisperModel("small", compute_type="int8")

embedder = get_embedder()
whisper_model = get_whisper_model()

# ---------------------- RAG Index Helpers ----------------------
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)
DOCS_STORE = os.path.join(INDEX_DIR, "docs.json")
FAISS_STORE = os.path.join(INDEX_DIR, "faiss.index")
EMB_DIM = 384  # matches the embedder above

def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def pdf_to_chunks(file_bytes, file_name):
    reader = PdfReader(io.BytesIO(file_bytes))
    all_chunks, meta = [], []
    for p, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        for ch in chunk_text(txt):
            all_chunks.append(ch)
            meta.append({"doc": file_name, "page": p+1})
    return all_chunks, meta

def build_or_extend_index(new_chunks, new_meta):
    # Load existing
    if os.path.exists(FAISS_STORE) and os.path.exists(DOCS_STORE):
        index = faiss.read_index(FAISS_STORE)
        with open(DOCS_STORE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # embed new
        new_embs = embedder.encode(new_chunks, convert_to_numpy=True)
        index.add(new_embs.astype("float32"))
        meta.extend(new_meta)
    else:
        index = faiss.IndexFlatIP(EMB_DIM)  # cosine via normalized
        embs = embedder.encode(new_chunks, convert_to_numpy=True)
        # normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = (embs / norms).astype("float32")
        index = faiss.IndexFlatIP(EMB_DIM)
        index.add(embs)
        meta = new_meta
    # Save
    faiss.write_index(index, FAISS_STORE)
    with open(DOCS_STORE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def retrieve(query, k=5):
    if not (os.path.exists(FAISS_STORE) and os.path.exists(DOCS_STORE)):
        return []
    index = faiss.read_index(FAISS_STORE)
    with open(DOCS_STORE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    q = embedder.encode([query], convert_to_numpy=True)
    q = (q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)).astype("float32")
    D, I = index.search(q, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        # guard if empty
        if idx == -1 or idx >= len(meta): 
            continue
        results.append({
            "score": float(score),
            "meta": meta[idx]
        })
    return results

# ---------------------- ASR ----------------------
def transcribe_audio(file_bytes, language="te"):  # 'te' = Telugu
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        segments, info = whisper_model.transcribe(tmp.name, language=language)
    text = " ".join(seg.text.strip() for seg in segments)
    return text

# ---------------------- LLM Prompts ----------------------
SUMMARY_SYSTEM_TE = (
    "మీరు పారాలీగల్ సహాయకుడు. ఇది న్యాయ సలహా కాదు. "
    "ఇచ్చిన సందర్భం ఆధారంగా మాత్రమే సమాధానం ఇవ్వండి. మూలాలు (Doc, Page) పేర్కొనండి."
)
SUMMARY_USER = (
    "క్రింది ట్రాన్స్క్రిప్ట్ మరియు డాక్యుమెంట్ స్నిపెట్ల ఆధారంగా కేసును తెలుగులో సారాంశం చేయండి. "
    "కీలక ఘటనలు, తేదీలు, పక్షాలు, పెండింగు పనులను బుల్లెట్లుగా ఇవ్వండి."
)

DRAFT_SYSTEM = (
    "You generate concise legal-adjacent drafts (not legal advice). "
    "Use only the provided transcript and context."
)

def call_llm(messages, model="gpt-4o-mini", temperature=0.2):
    if client is None:
        return "LLM key missing. Add your OPENAI_API_KEY in .env."
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages
    )
    return resp.choices[0].message.content.strip()

def build_context_snippets(results):
    # Turn retrieval results into text citations
    lines = []
    for r in results:
        doc = r["meta"]["doc"]; page = r["meta"]["page"]; score = r["score"]
        lines.append(f"(Doc: {doc}, Page: {page}, Score: {score:.2f})")
    return "\n".join(lines) if lines else "No sources."

# ---------------------- ICS helper ----------------------
def make_ics(title, when_dt, duration_minutes=30, description=""):
    cal = Calendar()
    e = Event()
    e.name = title
    e.begin = when_dt
    e.duration = datetime.timedelta(minutes=duration_minutes)
    e.description = description
    cal.events.add(e)
    return str(cal)

# ---------------------- UI ----------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Upload & Index", "🎙️ Voice → Transcript", "🔎 Q&A + Draft", "🗓️ Reminder"]
)

with tab1:
    st.subheader("Upload legal PDFs")
    uploaded = st.file_uploader("Drop PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        all_chunks, all_meta = [], []
        for f in uploaded:
            chunks, meta = pdf_to_chunks(f.read(), f.name)
            all_chunks += chunks
            all_meta += meta
        if st.button("Build / Update Index"):
            build_or_extend_index(all_chunks, all_meta)
            st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded)} file(s).")

with tab2:
    st.subheader("Telugu voice → transcript")
    audio = st.file_uploader("Upload .wav/.mp3", type=["wav", "mp3"])
    lang = st.selectbox("Language for ASR", ["te (Telugu)", "auto"], index=0)
    if audio and st.button("Transcribe"):
        transcript = transcribe_audio(audio.read(), language="te" if lang.startswith("te") else None)
        st.session_state["transcript"] = transcript
        st.success("Transcribed!")
        st.text_area("Transcript", value=transcript, height=200)

with tab3:
    st.subheader("Ask about the case (RAG)")
    q = st.text_input("Your question (Telugu or English)")
    k = st.slider("How many similar snippets?", 1, 8, 5)
    if st.button("Answer"):
        ctx_results = retrieve(q, k=k)
        srcs = build_context_snippets(ctx_results)
        transcript = st.session_state.get("transcript", "")
        # Simple prompt: (You can also pass actual snippet text. For MVP we pass citations & rely on user’s question)
        msgs = [
            {"role": "system", "content": SUMMARY_SYSTEM_TE},
            {"role": "user", "content": f"{SUMMARY_USER}\n\nTranscript:\n{transcript}\n\nQuestion:\n{q}\n\nSources:\n{srcs}"}
        ]
        ans = call_llm(msgs)
        st.markdown("**Answer:**")
        st.write(ans)
        st.markdown("**Sources (citations)**")
        st.code(srcs)

    st.markdown("---")
    st.subheader("Draft generator")
    draft_type = st.selectbox("Template", ["Statement of Facts", "Follow-up Email to Attorney"])
    if st.button("Generate Draft"):
        transcript = st.session_state.get("transcript", "")
        ctx_results = retrieve("case summary", k=5)
        srcs = build_context_snippets(ctx_results)
        template = "Write a concise Statement of Facts in Telugu." if draft_type.startswith("Statement") \
                   else "Draft a brief professional email (English) requesting case update and listing next steps."
        msgs = [
            {"role": "system", "content": DRAFT_SYSTEM},
            {"role": "user", "content": f"Template intent: {template}\n\nTranscript:\n{transcript}\n\nCitations:\n{srcs}\n\nAdd placeholders for missing details."}
        ]
        draft = call_llm(msgs, temperature=0.3)
        st.text_area("Draft", value=draft, height=280)

with tab4:
    st.subheader("Create a reminder (.ics)")
    title = st.text_input("Title", "Follow up on case")
    date = st.date_input("Date", datetime.date.today())
    time = st.time_input("Time", datetime.time(10, 0))
    duration = st.number_input("Duration (minutes)", 5, 240, 30)
    desc = st.text_area("Description (optional)", "")
    if st.button("Download .ics"):
        when = datetime.datetime.combine(date, time)
        ics_text = make_ics(title, when, duration, desc)
        st.download_button("Save .ics", data=ics_text, file_name="reminder.ics", mime="text/calendar")
        st.success("ICS ready!")
