"""Code-domain CTC forced aligner: word timings straight from IndexTTS2 semantic codes.

Self-contained copy of index-finetune/aligner/{ctc_model,align,pack_data}.py (STATE.md §16)
so the serving fork has no dependency on the training repo. CPU is enough: ~25-40 ms per
sentence at 4 threads (§16-J). Usage:

    aligner = CodeAligner("/path/to/code_aligner.pt")          # device="cpu"
    words = aligner.align(codes, "Hello there, viewer!")      # codes: 1-D int array, 50/s
    # -> one {"word","start","end","score","aligned"} per whitespace word, seconds from the
    #    start of the code sequence

Words that normalize to nothing (emotes, emoji, punctuation) are interpolated between their
aligned neighbours, so the count contract (one entry per whitespace word) always holds.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import torch
import torch.nn as nn

APOSTROPHES = "'’‘ʼ`´"
WORD_SEP = "|"
BLANK, UNK = "<b>", "<u>"


def normalize_word(word: str) -> str:
    """Casefold and keep only letters/marks/digits/apostrophe/hyphen. '' if nothing survives."""
    out = []
    for ch in unicodedata.normalize("NFKC", word).lower():
        if ch in APOSTROPHES:
            out.append("'")
        elif ch == "-" or ch == "‑":
            out.append("-")
        else:
            cat = unicodedata.category(ch)
            if cat[0] in ("L", "M", "N"):
                out.append(ch)
    s = "".join(out).strip("-'")
    return s



N_CODES = 8192
PAD_CODE = 8192      # padding index in the code embedding
MASK_CODE = 8193     # augmentation: dropped code
FRAME_SEC = 0.02


class CodeCTC(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 384, n_layers: int = 6, n_heads: int = 6,
                 ff: int = 1536, dropout: float = 0.1, max_len: int = 2048, conv_front: int = 0,
                 subsample: int = 1, rnn_layers: int = 0, upsample: int = 1):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        ff=ff, dropout=dropout, max_len=max_len, conv_front=conv_front, subsample=subsample,
                        rnn_layers=rnn_layers, upsample=upsample)
        self.subsample = subsample
        self.upsample = upsample
        self.frame_sec = FRAME_SEC * subsample / upsample
        # optional 2x time upsampling before the CTC head: cheap encoder at 40 ms, emissions at 20 ms
        self.up = nn.Conv1d(d_model, d_model, 3, padding=1) if upsample > 1 else None
        # optional local-context front-end: conv_front x [Conv1d(k=5, same) + GELU], residual
        self.conv = nn.ModuleList([nn.Conv1d(d_model, d_model, 5, padding=2) for _ in range(conv_front)])
        # optional time subsampling (stride-s conv, k=5, pad=2): T -> ceil(T/s); cuts the CTC blank ratio
        self.sub = nn.Conv1d(d_model, d_model, 5, stride=subsample, padding=2) if subsample > 1 else None
        self.emb = nn.Embedding(N_CODES + 2, d_model, padding_idx=PAD_CODE)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        if rnn_layers > 0:
            # bidirectional GRU encoder (Deep-Speech-2 style): reliable CTC convergence from scratch
            self.rnn = nn.GRU(d_model, d_model // 2, rnn_layers, batch_first=True, bidirectional=True,
                              dropout=dropout if rnn_layers > 1 else 0.0)
            self.enc = None
        else:
            self.rnn = None
            layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, codes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """codes [B,T] long (PAD_CODE padded), lengths [B] -> logits [B,T,V]."""
        B, T = codes.shape
        pos = torch.arange(T, device=codes.device).clamp(max=self.cfg["max_len"] - 1)
        x = self.emb(codes)
        if self.conv:
            keep = (torch.arange(T, device=codes.device)[None] < lengths[:, None]).unsqueeze(1).to(x.dtype)
            h = x.transpose(1, 2)
            for c in self.conv:
                h = h + torch.nn.functional.gelu(c(h * keep))
            x = h.transpose(1, 2)
        if self.sub is not None:
            keep = (torch.arange(T, device=codes.device)[None] < lengths[:, None]).unsqueeze(1).to(x.dtype)
            x = torch.nn.functional.gelu(self.sub(x.transpose(1, 2) * keep)).transpose(1, 2)
            lengths = self.out_lengths(lengths)
            T = x.shape[1]
            pos = torch.arange(T, device=codes.device).clamp(max=self.cfg["max_len"] - 1)
        if self.rnn is not None:
            x = self.drop(x)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.rnn(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            x = self.drop(x + self.pos(pos)[None])
            pad_mask = torch.arange(T, device=codes.device)[None] >= lengths[:, None]
            x = self.enc(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        if self.up is not None:
            x = x.repeat_interleave(self.upsample, dim=1)
            x = x + torch.nn.functional.gelu(self.up(x.transpose(1, 2))).transpose(1, 2)
        return self.head(x)

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return ((lengths - 1) // self.subsample + 1) * self.upsample

    @torch.no_grad()
    def log_probs(self, codes: torch.Tensor) -> torch.Tensor:
        """Single utterance: codes [T] -> log-probs [T,V] (float32)."""
        c = codes.long().unsqueeze(0)
        logits = self.forward(c, torch.tensor([c.shape[1]], device=c.device))
        return torch.log_softmax(logits[0].float(), dim=-1)


def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg.setdefault("conv_front", 0)
    cfg.setdefault("subsample", 1)
    cfg.setdefault("rnn_layers", 0)
    cfg.setdefault("upsample", 1)
    model = CodeCTC(**cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck["symbols"]


NEG = -1e30


def ctc_viterbi(log_probs: torch.Tensor, targets: list[int], blank: int = 0):
    """Best monotonic CTC path. log_probs [T,V]; returns per-target (start, end) frame
    spans (end exclusive) and mean per-token log-prob, or None if infeasible."""
    T = log_probs.shape[0]
    L = len(targets)
    S = 2 * L + 1
    ext = torch.full((S,), blank, dtype=torch.long, device=log_probs.device)
    ext[1::2] = torch.tensor(targets, dtype=torch.long, device=log_probs.device)
    # skip allowed from s-2 when ext[s] is a label different from ext[s-2]
    skip = torch.zeros(S, dtype=torch.bool, device=log_probs.device)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    emit = log_probs[:, ext]  # [T,S]
    trellis = torch.full((T, S), NEG, device=log_probs.device)
    back = torch.zeros((T, S), dtype=torch.int8, device=log_probs.device)  # 0 stay,1 from s-1,2 from s-2
    trellis[0, 0] = emit[0, 0]
    if S > 1:
        trellis[0, 1] = emit[0, 1]
    for t in range(1, T):
        prev = trellis[t - 1]
        c0 = prev
        c1 = torch.cat([prev.new_full((1,), NEG), prev[:-1]])
        c2 = torch.cat([prev.new_full((2,), NEG), prev[:-2]])
        c2 = torch.where(skip, c2, torch.full_like(c2, NEG))
        stacked = torch.stack([c0, c1, c2])
        best, arg = stacked.max(0)
        trellis[t] = best + emit[t]
        back[t] = arg.to(torch.int8)
    # end in last blank or last label
    end_states = [S - 1] if S == 1 else [S - 1, S - 2]
    fin = trellis[T - 1, end_states]
    if fin.max() <= NEG / 2:
        return None
    s = end_states[int(fin.argmax())]
    path = torch.empty(T, dtype=torch.long)
    for t in range(T - 1, -1, -1):
        path[t] = s
        if t:
            s -= int(back[t, s])
    path = path.tolist()
    spans = [None] * L
    scores = [[] for _ in range(L)]
    for t, st in enumerate(path):
        if st % 2 == 1:
            i = st // 2
            if spans[i] is None:
                spans[i] = [t, t + 1]
            else:
                spans[i][1] = t + 1
            scores[i].append(float(log_probs[t, targets[i]]))
    return [(a, b, sum(sc) / max(1, len(sc))) for (a, b), sc in zip(spans, scores)]


class CodeAligner:
    def __init__(self, ckpt_path: str, device: str = "cpu"):
        self.model, self.symbols = load_model(ckpt_path, device)
        self.sym2id = {s: i for i, s in enumerate(self.symbols)}
        self.device = device

    def _targets(self, words: list[str]):
        """Return (targets, per-word (first_idx, n_tokens)) using WORD_SEP between alignable words."""
        targets, spans = [], []
        first = True
        for w in words:
            norm = normalize_word(w)
            if not norm:
                spans.append(None)
                continue
            if not first:
                targets.append(self.sym2id[WORD_SEP])
            first = False
            start = len(targets)
            targets.extend(self.sym2id.get(c, self.sym2id[UNK]) for c in norm)
            spans.append((start, len(norm)))
        return targets, spans

    def align(self, codes, text: str) -> list[dict]:
        words = text.split()
        if not words:
            return []
        codes_t = torch.as_tensor(np.asarray(codes), dtype=torch.long, device=self.device)
        T = int(codes_t.shape[0])
        total = T * FRAME_SEC
        fs = self.model.frame_sec
        targets, spans = self._targets(words)
        result = [None] * len(words)
        if targets:
            lp = self.model.log_probs(codes_t)
            tok = ctc_viterbi(lp, targets)
            if tok is not None:
                for wi, sp in enumerate(spans):
                    if sp is None:
                        continue
                    start, n = sp
                    a = tok[start][0]
                    b = tok[start + n - 1][1]
                    score = sum(tok[start + k][2] for k in range(n)) / n
                    result[wi] = {"word": words[wi], "start": min(a * fs, total), "end": min(b * fs, total),
                                  "score": score, "aligned": True}
        return fill_unaligned(words, result, total)


def fill_unaligned(words, result, total):
    """Interpolate missing words evenly inside the gap between aligned neighbours."""
    n = len(words)
    i = 0
    while i < n:
        if result[i] is not None:
            i += 1
            continue
        j = i
        while j < n and result[j] is None:
            j += 1
        t0 = result[i - 1]["end"] if i > 0 else 0.0
        t1 = result[j]["start"] if j < n else total
        if t1 < t0:
            t1 = t0
        k = j - i
        for m in range(k):
            a = t0 + (t1 - t0) * m / k
            b = t0 + (t1 - t0) * (m + 1) / k
            result[i + m] = {"word": words[i + m], "start": a, "end": b, "score": None, "aligned": False}
        i = j
    return result


