"""
Lightweight BM25Okapi over tokenized documents (no extra dependencies).
"""
from __future__ import annotations

import math
import re
from typing import List, Sequence


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", (text or "").lower())


class BM25Okapi:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [list(d) for d in corpus_tokens]
        self.N = len(self.corpus)
        if self.N == 0:
            self.avgdl = 0.0
            self.df = {}
            self.idf = {}
            return
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / self.N
        df = {}
        for doc in self.corpus:
            seen = set(doc)
            for w in seen:
                df[w] = df.get(w, 0) + 1
        self.idf = {}
        for w, freq in df.items():
            # BM25 idf variant (robust for short queries)
            self.idf[w] = math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> List[float]:
        if self.N == 0:
            return []
        q = list(query_tokens)
        scores = [0.0] * self.N
        for i, doc in enumerate(self.corpus):
            if not doc:
                continue
            dl = self.doc_len[i]
            freqs: dict[str, int] = {}
            for w in doc:
                freqs[w] = freqs.get(w, 0) + 1
            score = 0.0
            for w in set(q):
                if w not in freqs:
                    continue
                idf = self.idf.get(w)
                if idf is None:
                    continue
                f = freqs[w]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl > 0 else f + self.k1
                score += idf * (f * (self.k1 + 1)) / denom
            scores[i] = score
        return scores
