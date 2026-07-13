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

# Point the app at the checkpoint (any ONE of these):
#   a) drop it here:            ./model/best_pashto_sft.pt
#   b) or set a path:           export MODEL_PATH=/full/path/best_pashto_sft.pt
#   c) or pull from HF Hub:     export HF_REPO_ID=<user>/pashto-slm-chat

streamlit run app.py
```

The app opens at http://localhost:8501.

---

## Deploy for free (very easy to share)

1. Push this repo to GitHub (already done if you cloned it).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**.
3. Repo `iaamshayan/pashto-streamlit-nlp`, branch `main`, main file `app.py`.
4. In **Advanced settings → Secrets**, add the model location:

   ```toml
   HF_REPO_ID = "iaamshayan/pashto-slm-chat"
   # HF_FILENAME = "best_pashto_sft.pt"   # default
   # HF_TOKEN = "hf_..."                   # only if the repo is private
   ```

5. Deploy → you get a public `https://<app>.streamlit.app` URL to share.

The 382 MB checkpoint is **not** committed to git (it exceeds GitHub's file
limit); it is downloaded from the Hugging Face Hub on first run and cached.

### Hosting the checkpoint on the Hugging Face Hub (one time)

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create pashto-slm-chat --type model
huggingface-cli upload iaamshayan/pashto-slm-chat \
    "path/to/best_pashto_sft.pt" best_pashto_sft.pt
```

Then set `HF_REPO_ID = "iaamshayan/pashto-slm-chat"` as above.

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
