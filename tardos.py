from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    user_key: str


def _sha256_int(*parts: object, nbytes: int = 8) -> int:
    msg = "||".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(msg).digest()
    return int.from_bytes(digest[:nbytes], "big", signed=False)


def sample_tardos_probabilities(length: int, cutoff: float = 0.08, seed: int | str = 2026) -> np.ndarray:
    """Sample the Tardos bias vector p using a truncated arcsine distribution."""
    if not (0.0 < cutoff < 0.5):
        raise ValueError("cutoff must be in (0, 0.5)")
    rng = np.random.default_rng(_sha256_int("p", seed))
    u = rng.random(length)
    p = np.sin(0.5 * np.pi * u) ** 2
    p = cutoff + (1.0 - 2.0 * cutoff) * p
    return p.astype(np.float32)


def user_seed(master_key: str, user: UserRecord) -> int:
    digest = hmac.new(
        master_key.encode("utf-8"),
        f"{user.user_id}||{user.user_key}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_codebook(users: Sequence[UserRecord], p: np.ndarray, master_key: str) -> np.ndarray:
    """Generate a user-bound probabilistic fingerprint matrix.

    Each user's codeword is sampled with the same Tardos bias vector p but
    from a user-specific RNG seeded by HMAC(master_key, user_id || user_key).
    """
    codebook = np.zeros((len(users), len(p)), dtype=np.uint8)
    for idx, user in enumerate(users):
        rng = np.random.default_rng(user_seed(master_key, user))
        codebook[idx] = (rng.random(len(p)) < p).astype(np.uint8)
    return codebook


def accuse_symmetric(extracted_bits: np.ndarray, codebook: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Symmetric Tardos score under a binary observation model."""
    y = np.asarray(extracted_bits, dtype=np.uint8).reshape(1, -1)
    c = np.asarray(codebook, dtype=np.uint8)
    p = np.asarray(p, dtype=np.float64).reshape(1, -1)
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)

    plus_if_one = np.sqrt((1.0 - p) / p)
    plus_if_zero = np.sqrt(p / (1.0 - p))

    scores = np.zeros(c.shape, dtype=np.float64)
    mask_y1 = y == 1
    mask_y0 = ~mask_y1

    scores += np.where(mask_y1 & (c == 1), plus_if_one, 0.0)
    scores += np.where(mask_y1 & (c == 0), -plus_if_zero, 0.0)
    scores += np.where(mask_y0 & (c == 0), plus_if_zero, 0.0)
    scores += np.where(mask_y0 & (c == 1), -plus_if_one, 0.0)
    return scores.sum(axis=1)


def rank_suspects(users: Sequence[UserRecord], scores: np.ndarray) -> list[tuple[str, float]]:
    pairs = [(users[i].user_id, float(scores[i])) for i in range(len(users))]
    return sorted(pairs, key=lambda x: x[1], reverse=True)
