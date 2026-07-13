"""Inference wrapper for the Pashto chat SLM.

Loads the SFT checkpoint (ModernGPT: RoPE + RMSNorm + SwiGLU + GQA, ~100M) and
the 32k Metaspace BPE tokenizer, then generates a reply for a user turn using
the exact chat protocol the model was fine-tuned with:

    [CLS] <user tokens> [MASK] <assistant tokens> [SEP]

Loss during SFT was only on the assistant side; generation starts right after
[MASK] and stops at [SEP]. Special-token ids come from the checkpoint's
config (cls=1, mask=4, sep=2, pad=3) and are re-derived from the tokenizer.

Model resolution order (first that works wins):
  1. env / st.secrets  MODEL_PATH   -> a local .pt file
  2. ./model/best_pashto_sft.pt     -> bundled locally (gitignored)
  3. HF Hub download   HF_REPO_ID (+ optional HF_FILENAME) via huggingface_hub

Keep the generation defaults in sync with pashto-slm-sft.ipynb.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model_modern import ModernGPT, ModernGPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKENIZER = os.path.join(HERE, "tokenizer", "tokenizer.json")
DEFAULT_LOCAL_CKPT = os.path.join(HERE, "model", "best_pashto_sft.pt")
DEFAULT_HF_FILENAME = "best_pashto_sft.pt"


def _secret(name: str, default: str | None = None) -> str | None:
    """Read a value from st.secrets if available, else the environment."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def resolve_checkpoint() -> str:
    """Find the checkpoint file, downloading from HF Hub if needed."""
    explicit = _secret("MODEL_PATH")
    if explicit and os.path.exists(explicit):
        return explicit
    if os.path.exists(DEFAULT_LOCAL_CKPT):
        return DEFAULT_LOCAL_CKPT

    repo_id = _secret("HF_REPO_ID")
    if repo_id:
        from huggingface_hub import hf_hub_download

        filename = _secret("HF_FILENAME", DEFAULT_HF_FILENAME)
        token = _secret("HF_TOKEN")  # only needed for private repos
        return hf_hub_download(repo_id=repo_id, filename=filename, token=token)

    raise FileNotFoundError(
        "No model checkpoint found. Set MODEL_PATH to a local .pt, place it at "
        f"{DEFAULT_LOCAL_CKPT}, or set HF_REPO_ID (and HF_FILENAME) so it can be "
        "downloaded from the Hugging Face Hub."
    )


class PashtoChatModel:
    """Loaded model + tokenizer + chat protocol."""

    def __init__(self, ckpt_path: str, tokenizer_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enc = Tokenizer.from_file(tokenizer_path)

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(ck["config"])
        fields = ModernGPTConfig.__dataclass_fields__
        self.config = ModernGPTConfig(**{k: v for k, v in cfg.items() if k in fields})
        self.model = ModernGPT(self.config)

        sd = ck["model"]
        sd = {k[len("module."):] if k.startswith("module.") else k: v
              for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "rope" not in k]
        real_unexpected = [k for k in unexpected if "rope" not in k]
        if real_missing or real_unexpected:
            raise RuntimeError(
                f"state dict mismatch: missing={real_missing}, "
                f"unexpected={real_unexpected}")
        self.model.eval().to(self.device)

        # special-token ids (already in the 32k vocab; no vocab change for chat)
        self.CLS_ID = self.enc.token_to_id("[CLS]")
        self.MASK_ID = self.enc.token_to_id("[MASK]")
        self.SEP_ID = self.enc.token_to_id("[SEP]")
        self.PAD_ID = self.enc.token_to_id("[PAD]")
        assert None not in (self.CLS_ID, self.MASK_ID, self.SEP_ID, self.PAD_ID)

        self.step = ck.get("step")
        self.val_loss = ck.get("val_loss")
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def _generate(self, idx, max_new_tokens, temperature, top_k, top_p,
                  rep_penalty, no_repeat_ngram):
        """Ported verbatim from generate_chat() in pashto-slm-sft.ipynb."""
        m = self.model
        stop_id = self.SEP_ID
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= m.config.block_size \
                else idx[:, -m.config.block_size:]
            logits, _ = m(idx_cond)
            logits = logits[:, -1, :].float()
            if rep_penalty and rep_penalty != 1.0:
                for t in set(idx[0].tolist()):
                    logits[0, t] = logits[0, t] / rep_penalty \
                        if logits[0, t] > 0 else logits[0, t] * rep_penalty
            if no_repeat_ngram > 0 and idx.size(1) >= no_repeat_ngram:
                seq = idx[0].tolist()
                n = no_repeat_ngram
                seen = {}
                for i in range(len(seq) - n + 1):
                    seen.setdefault(tuple(seq[i:i + n - 1]), set()).add(seq[i + n - 1])
                for b in seen.get(tuple(seq[-(n - 1):]), set()):
                    logits[0, b] = -float("inf")
            logits = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p and top_p < 1.0:
                s, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(s, dim=-1), dim=-1)
                keep_mask = cum > top_p
                keep_mask[:, 1:] = keep_mask[:, :-1].clone()
                keep_mask[:, 0] = False
                logits[0, si[0][keep_mask[0]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat((idx, nxt), dim=1)
            if nxt.item() == stop_id:
                break
        return idx

    def chat(self, prompt: str, max_new_tokens=160, temperature=0.8, top_k=50,
             top_p=0.95, rep_penalty=1.3, no_repeat_ngram=3) -> str:
        """[CLS] prompt [MASK] -> generate until [SEP]; return the reply text."""
        ids = [self.CLS_ID] + self.enc.encode(prompt).ids + [self.MASK_ID]
        idx = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self._generate(idx, max_new_tokens, temperature, top_k, top_p,
                             rep_penalty, no_repeat_ngram)
        reply = out[0].tolist()[len(ids):]
        if self.SEP_ID in reply:
            reply = reply[:reply.index(self.SEP_ID)]
        return self.enc.decode(reply).strip()
