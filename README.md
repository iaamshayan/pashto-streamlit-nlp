# پښتو چ‌ټ · Pashto Chat

A shareable **Streamlit chat app** for a Pashto small language model trained
**from scratch** — a ~100M-param decoder-only transformer (RoPE + RMSNorm +
SwiGLU + grouped-query attention) pretrained on a ~644M-word Pashto corpus and
then chat-fine-tuned (SFT) with a masked-loss instruction dataset.

> ⚠️ Research model. It writes Pashto script only — no punctuation, digits or
> Latin (stripped from the corpus by design) — and can be factually wrong. It
> has no access to the time, date or live news.

---

## Quick start (local)

```bash
git clone https://github.com/iaamshayan/pashto-streamlit-nlp.git
cd pashto-streamlit-nlp
pip install -r requirements.txt

streamlit run app.py
```

The app opens at http://localhost:8501. On first run it downloads the 382 MB
checkpoint from the Hugging Face Hub
([`iaamshayan/pashto-slm-chat`](https://huggingface.co/iaamshayan/pashto-slm-chat))
and caches it. To use a local file instead, drop it at `./model/best_pashto_sft.pt`
or set `export MODEL_PATH=/full/path/best_pashto_sft.pt`.

---

## Deploy for free (very easy to share)

1. Push this repo to GitHub (already done if you cloned it).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**.
3. Repo `iaamshayan/pashto-streamlit-nlp`, branch `main`, main file `app.py`.
4. **Deploy.** No secrets needed — the app already points at the public model
   repo [`iaamshayan/pashto-slm-chat`](https://huggingface.co/iaamshayan/pashto-slm-chat)
   and downloads the checkpoint on first run.

You get a public `https://<app>.streamlit.app` URL to share.

The 382 MB checkpoint is **not** committed to git (it exceeds GitHub's file
limit); it is downloaded from the Hugging Face Hub on first run and cached.

### Overriding the model source (optional)

Point at a different / private checkpoint via **Advanced settings → Secrets**
(or local env vars):

```toml
HF_REPO_ID  = "your-user/your-model"   # default: iaamshayan/pashto-slm-chat
HF_FILENAME = "best_pashto_sft.pt"     # default
HF_TOKEN    = "hf_..."                 # only if the repo is private
```

---

## How it works

The model was fine-tuned with a fixed chat protocol using special tokens that
already exist in the 32k BPE vocab (no vocab change):

```
[CLS] <user tokens> [MASK] <assistant tokens> [SEP]
```

At inference the app builds `[CLS] prompt [MASK]`, samples tokens (temperature /
top-k / top-p / repetition penalty / no-repeat-ngram — all tunable in the
sidebar) and stops at `[SEP]`. See `inference.py`.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat UI (RTL, example prompts, sampling controls) |
| `inference.py` | Checkpoint loading + chat generation |
| `model_modern.py` | The ModernGPT architecture (must match the checkpoint) |
| `tokenizer/tokenizer.json` | 32k Metaspace BPE tokenizer (reversible decode) |
| `requirements.txt` | CPU-only dependencies |

## Model card (short)

- **Architecture:** ModernGPT — 12 layers, 12 heads, d=768, GQA (4 KV heads),
  RoPE, RMSNorm, SwiGLU, tied embeddings, block size 512.
- **Params:** ~100M · **SFT val loss:** ~2.82.
- **Data:** Pashto-script-only corpus (~1.13M docs / ~644M words) + a masked-loss
  Pashto instruction set.
- Part of the [Pashto-NLP](https://github.com/iaamshayan/Pashto-NLP) research
  stack (corpus builder, tokenizers, SLM, SFT builder).
