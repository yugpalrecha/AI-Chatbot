import streamlit as st
import requests
import uuid
import os

BACKEND_URL = os.getenv("BACKEND_URL")

# ────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────
def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    new_thread = generate_thread_id()
    st.session_state.thread_id = new_thread
    if new_thread not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(new_thread)
    st.query_params["thread_id"] = new_thread
    st.query_params["all_threads"] = ",".join(st.session_state.chat_threads)


def fetch_history(thread_id):
    try:
        res = requests.get(f"{BACKEND_URL}/history/{thread_id}", timeout=5)
        return res.json().get("messages", [])
    except Exception:
        return []


def get_rag_status():
    try:
        res = requests.get(f"{BACKEND_URL}/rag-status", timeout=3)
        return res.json()
    except Exception:
        return {"ready": False, "filename": None}


# ────────────────────────────────────────────
#  PAGE CONFIG
# ────────────────────────────────────────────
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

# ────────────────────────────────────────────
#  SESSION STATE INIT (survives refresh via URL)
# ────────────────────────────────────────────
params = st.query_params

if "chat_threads" not in st.session_state:
    saved = params.get("all_threads", "")
    st.session_state.chat_threads = saved.split(",") if saved else []

if "thread_id" not in st.session_state:
    saved_thread = params.get("thread_id", "")
    if saved_thread:
        st.session_state.thread_id = saved_thread
        if saved_thread not in st.session_state.chat_threads:
            st.session_state.chat_threads.append(saved_thread)
    else:
        new_thread = generate_thread_id()
        st.session_state.thread_id = new_thread
        st.session_state.chat_threads.append(new_thread)
        st.query_params["thread_id"] = new_thread
        st.query_params["all_threads"] = new_thread

if "upload_msg" not in st.session_state:
    st.session_state.upload_msg = None

# ────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.divider()

    # ── PDF UPLOAD ──────────────────────────
    st.subheader("📄 Upload PDF for Q&A")

    rag_status = get_rag_status()

    if rag_status.get("ready"):
        st.success(f"✅ Active: **{rag_status['filename']}**")

        if st.button("❌ Cancel PDF", use_container_width=True):
            try:
                requests.post(f"{BACKEND_URL}/clear-pdf", timeout=5)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("No PDF loaded yet.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.button("📤 Upload & Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=60,
                    )
                    data = response.json()

                    if response.ok:
                        st.session_state.upload_msg = (
                            "success",
                            data.get("message", "PDF uploaded!")
                        )
                    else:
                        st.session_state.upload_msg = (
                            "error",
                            data.get("error", "Upload failed")
                        )

                except Exception as e:
                    st.session_state.upload_msg = ("error", str(e))

            st.rerun()

    if st.session_state.upload_msg:
        kind, msg = st.session_state.upload_msg
        if kind == "success":
            st.success(msg)
        else:
            st.error(msg)
        st.session_state.upload_msg = None

    st.divider()

    # ── CONVERSATIONS ───────────────────────
    st.subheader("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    for tid in reversed(st.session_state.chat_threads):
        label = f"🗨 Chat {tid[:8]}..."
        is_active = tid == st.session_state.thread_id
        btn_type = "primary" if is_active else "secondary"

        if st.button(label, key=f"btn_{tid}", use_container_width=True, type=btn_type):
            st.session_state.thread_id = tid
            st.query_params["thread_id"] = tid
            st.query_params["all_threads"] = ",".join(st.session_state.chat_threads)
            st.rerun()

    st.divider()
    st.caption("Powered by Groq + LangGraph")

# ────────────────────────────────────────────
#  MAIN CHAT AREA
# ────────────────────────────────────────────
st.title("🤖 AI Assistant")
st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")

# Fetch and display history
messages = fetch_history(st.session_state.thread_id)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ────────────────────────────────────────────
#  CHAT INPUT
# ────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        try:
            with requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": user_input,
                    "thread_id": st.session_state.thread_id,
                },
                stream=True,
                timeout=60,
            ) as response:
                for chunk in response.iter_content(chunk_size=32):
                    if chunk:
                        text = chunk.decode("utf-8", errors="ignore")
                        full_reply += text
                        placeholder.markdown(full_reply + "▌")

            placeholder.markdown(full_reply)

        except requests.exceptions.ConnectionError:
            placeholder.error("❌ Cannot connect to backend. Is `node backend.js` running?")
        except Exception as e:
            placeholder.error(f"❌ Error: {e}")

    # Keep thread list in URL
    st.query_params["all_threads"] = ",".join(st.session_state.chat_threads)