"""Synchronous inference engine. Mirrors research_pipeline.py (the Colab cells) exactly:
same repos, max_len 64, label-name fallback, Qwen chat template with the DEFAULT system message,
'{"topic": "' prefill, same stop sequences, temperature 0.

Used by (a) ml_service.py (Colab model server) and (b) ml.local.LocalMLRuntime (in-process).
Heavy imports (torch / transformers / llama_cpp) happen inside load(), never at import time.
"""
from __future__ import annotations

import os
import threading

INSTRUCTION = ("Extract a normalized search topic and a one-sentence paraphrase "
               "from the student's request.")
PREFILL = '{"topic": "'
STOPS = ['", "brief"', '"}', "<|im_end|>"]
FALLBACK_LABELS = ["fetch_sources", "create_report", "chat", "unclear"]
MAX_INPUT_CHARS = 600   # n_ctx=512 tokens: keep prompt + input safely inside the context window


def build_prompt(text: str) -> str:
    # Must match training: Qwen chat template including the default system message.
    return ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
            "You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{INSTRUCTION}\n\nRequest: {text[:MAX_INPUT_CHARS]}<|im_end|>\n"
            "<|im_start|>assistant\n")


class Engine:
    CLF_VERSION = "modernbert_v1"
    EXT_VERSION = "extractor_v1_q4km_topic_only"

    def __init__(self, env=None):
        e = env if env is not None else os.environ
        hf_user = e.get("HF_USER", "user")
        self.clf_repo = e.get("HF_CLASSIFIER_REPO", f"{hf_user}/research-intent-clf-v1")
        self.gguf_repo = e.get("HF_EXTRACTOR_GGUF_REPO", f"{hf_user}/research-extractor-qwen1.5b-gguf-v1")
        self.gguf_name = e.get("HF_EXTRACTOR_GGUF_FILE", "qwen-extractor-q4_k_m.gguf")
        # Drive copy is used first if it exists (Colab); otherwise the Hub repo is used.
        self.drive_gguf = e.get("DRIVE_GGUF",
                                "/content/drive/MyDrive/research_assistant-1/gguf/qwen-extractor-q4_k_m.gguf")
        self.clf_max_len = int(e.get("CLF_MAX_LEN", "64"))
        self.n_threads = int(e.get("N_THREADS", os.cpu_count() or 2))
        self.want_logprobs = e.get("EXTRACTOR_LOGPROBS", "false").lower() in ("1", "true", "yes")
        self.ready = False
        self.info: dict = {}
        self._llm_lock = threading.Lock()   # llama.cpp instances are NOT thread-safe

    # ---------------- loading ----------------
    @staticmethod
    def _hf_login() -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get("HF_TOKEN")
            except Exception:
                token = None
        if token:
            from huggingface_hub import login
            login(token=token)

    def load(self) -> None:
        import torch
        from llama_cpp import Llama
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._hf_login()
        torch.set_num_threads(self.n_threads)

        self.tok = AutoTokenizer.from_pretrained(self.clf_repo)
        self.clf = AutoModelForSequenceClassification.from_pretrained(self.clf_repo).eval()
        id2label = {int(k): v for k, v in self.clf.config.id2label.items()}
        self.label_fallback_used = all(v.startswith("LABEL_") for v in id2label.values())
        if self.label_fallback_used:
            id2label = dict(enumerate(FALLBACK_LABELS))   # config had no names: use training order
        self.id2label = id2label

        if os.path.exists(self.drive_gguf):
            gguf_path = self.drive_gguf
        else:
            from huggingface_hub import hf_hub_download
            gguf_path = hf_hub_download(self.gguf_repo, self.gguf_name)
        kw = {"logits_all": True} if self.want_logprobs else {}
        self.llm = Llama(model_path=gguf_path, n_ctx=512, n_threads=self.n_threads,
                         n_batch=128, verbose=False, **kw)
        self.info = {"classifier_repo": self.clf_repo, "gguf_path": os.path.basename(gguf_path),
                     "labels": [self.id2label[i] for i in sorted(self.id2label)],
                     "label_fallback_used": self.label_fallback_used, "n_threads": self.n_threads,
                     "logprobs": self.want_logprobs,
                     "clf_version": self.CLF_VERSION, "ext_version": self.EXT_VERSION}
        self.ready = True

    def warmup(self) -> None:
        self.classify("warm up")
        self.extract_topic("warm up")

    # ---------------- stages ----------------
    def classify(self, text: str) -> tuple[str, float, dict[str, float]]:
        import torch
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.clf_max_len)
        with torch.no_grad():
            probs = torch.softmax(self.clf(**enc).logits, -1)[0]
        scores = {self.id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        i = int(probs.argmax())
        return self.id2label[i], round(float(probs[i]), 4), scores

    def extract_topic(self, text: str, max_tokens: int = 32) -> tuple[str, float | None]:
        prompt = build_prompt(text) + PREFILL
        kw = {"logprobs": 1} if self.want_logprobs else {}
        with self._llm_lock:
            try:
                out = self.llm(prompt, max_tokens=max_tokens, temperature=0.0, stop=STOPS, **kw)
            except ValueError:               # logprobs unsupported in this build: degrade quietly
                self.want_logprobs, kw = False, {}
                out = self.llm(prompt, max_tokens=max_tokens, temperature=0.0, stop=STOPS)
        choice = out["choices"][0]
        topic = choice["text"].strip().strip('"').strip()
        avg_lp = None
        if kw:
            tl = [x for x in ((choice.get("logprobs") or {}).get("token_logprobs") or []) if x is not None]
            avg_lp = round(sum(tl) / len(tl), 4) if tl else None
        return topic, avg_lp
