"""Pashto Chat — a Streamlit chat UI for the from-scratch Pashto SLM.

A ~100M-param decoder-only model (ModernGPT: RoPE + RMSNorm + SwiGLU + GQA)
trained from scratch on a ~644M-word Pashto corpus, then chat-fine-tuned (SFT)
with a masked-loss instruction dataset. This app wraps it in a shareable chat.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io -> point at app.py
"""
from __future__ import annotations

import os

import streamlit as st

from inference import PashtoChatModel, resolve_checkpoint, DEFAULT_TOKENIZER

st.set_page_config(page_title="پښتو چ‌ټ · Pashto Chat", page_icon="🗨️",
                   layout="centered")

# ---- right-to-left styling for Pashto text -------------------------------
st.markdown(
    """
    <style>
      .stChatMessage p, .stChatMessage li { direction: rtl; text-align: right;
          font-size: 1.12rem; line-height: 2.0; }
      textarea, input { direction: rtl; text-align: right; }
      h1 { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="ماډل لوډ کیږي… (loading model)")
def load_model() -> PashtoChatModel:
    ckpt = resolve_checkpoint()
    return PashtoChatModel(ckpt, DEFAULT_TOKENIZER)


st.title("پښتو چ‌ټ")
st.caption("A from-scratch ~100M Pashto language model · trained on a 644M-word "
           "corpus · chat fine-tuned")

# ---- sidebar: generation controls + model info ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    max_new = st.slider("Max new tokens", 32, 320, 160, 16)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8, 0.05)
    top_k = st.slider("Top-k", 0, 200, 50, 5)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.3, 0.05)
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "**Note:** this is a small research model trained from scratch on "
        "Pashto only. It has no punctuation, digits or Latin (stripped by "
        "design) and can be factually wrong. It cannot access the time, date "
        "or news."
    )

# ---- example prompts -----------------------------------------------------
EXAMPLES = [
    "د پښتو ژبې په اړه معلومات راکړه",
    "د ښې روغتیا لپاره څه وکړو",
    "یوه لنډه کیسه ولیکه",
    "ولې زده کړه مهمه ده",
]

try:
    model = load_model()
except Exception as e:  # pragma: no cover - surfaced to the user in the UI
    st.error("Could not load the model checkpoint.\n\n"
             "Set `HF_REPO_ID` in Streamlit secrets (or `MODEL_PATH` locally) "
             "so the app can find `best_pashto_sft.pt`. See the README.")
    st.exception(e)
    st.stop()

with st.sidebar:
    st.divider()
    st.caption(f"Params: **{model.n_params/1e6:.0f}M**  ·  "
               f"val loss: **{model.val_loss:.3f}**  ·  "
               f"device: **{model.device}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# example chips only before the first message
if not st.session_state.messages:
    st.write("Try one of these:")
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# ---- replay history ------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("… پوښتنه دلته ولیکئ  (type your question)")
prompt = prompt or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("… فکر کوم"):
            reply = model.chat(
                prompt, max_new_tokens=max_new, temperature=temperature,
                top_k=top_k or None, top_p=top_p, rep_penalty=rep_penalty)
        reply = reply or "…"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
