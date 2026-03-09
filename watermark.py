from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np
import pywt


@dataclass
class EmbedResult:
    image: np.ndarray
    logical_bits: np.ndarray
    extracted_preview: np.ndarray | None = None


class BlindDwtDctQim:
    """Blind watermarking with DWT + block-DCT + scalar QIM.

    - Blind extraction: original image is not required.
    - Keyed placement: selected blocks and coefficient signs depend on master_key + content_id.
    - Repetition code: each logical bit is embedded in multiple blocks and decoded by majority vote.
    """

    def __init__(
        self,
        master_key: str,
        wavelet: str = "haar",
        block_size: int = 8,
        coeff_pos: tuple[int, int] = (3, 3),
        delta: float = 26.0,
        repeats: int = 6,
        bands: tuple[str, ...] = ("LL", "LH"),
    ) -> None:
        self.master_key = master_key
        self.wavelet = wavelet
        self.block_size = block_size
        self.coeff_pos = coeff_pos
        self.delta = float(delta)
        self.repeats = int(repeats)
        self.bands = bands

    def _seed(self, *parts: object) -> int:
        msg = "||".join(map(str, (self.master_key, *parts))).encode("utf-8")
        return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big", signed=False)

    @staticmethod
    def _qim_embed(x: float, bit: int, delta: float) -> float:
        if int(bit) == 0:
            return round(x / delta) * delta
        return round((x - delta / 2.0) / delta) * delta + delta / 2.0

    @staticmethod
    def _qim_detect(x: float, delta: float) -> tuple[int, float]:
        q0 = round(x / delta) * delta
        q1 = round((x - delta / 2.0) / delta) * delta + delta / 2.0
        d0 = abs(x - q0)
        d1 = abs(x - q1)
        return int(d1 < d0), float(d0 - d1)

    def _split_bands(self, image_bgr: np.ndarray):
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y = ycrcb[:, :, 0]
        coeffs = pywt.dwt2(y, self.wavelet)
        ll, (lh, hl, hh) = coeffs
        return ycrcb, ll, lh, hl, hh

    def _merge_bands(self, ycrcb: np.ndarray, ll, lh, hl, hh) -> np.ndarray:
        y_rec = pywt.idwt2((ll, (lh, hl, hh)), self.wavelet)
        h, w = ycrcb.shape[:2]
        y_rec = y_rec[:h, :w]
        out = ycrcb.copy()
        out[:, :, 0] = y_rec
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

    def _candidate_positions(self, band_map: dict[str, np.ndarray]) -> list[tuple[str, int, int]]:
        positions: list[tuple[str, int, int]] = []
        for band_name in self.bands:
            band = band_map[band_name]
            h, w = band.shape
            bh = h // self.block_size
            bw = w // self.block_size
            for by in range(bh):
                for bx in range(bw):
                    positions.append((band_name, by, bx))
        return positions

    def _select_positions(self, band_map: dict[str, np.ndarray], n_bits: int, content_id: str) -> list[tuple[str, int, int]]:
        need = n_bits * self.repeats
        positions = self._candidate_positions(band_map)
        if need > len(positions):
            raise ValueError(f"Not enough capacity: need {need} blocks, only {len(positions)} available.")
        rng = np.random.default_rng(self._seed("positions", content_id, tuple((k, v.shape) for k, v in sorted(band_map.items())), n_bits))
        rng.shuffle(positions)
        return positions[:need]

    def _signs(self, count: int, content_id: str) -> np.ndarray:
        rng = np.random.default_rng(self._seed("signs", content_id, count))
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=count)

    def embed(self, image_bgr: np.ndarray, bits: np.ndarray, content_id: str) -> EmbedResult:
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        ycrcb, ll, lh, hl, hh = self._split_bands(image_bgr)
        band_map = {"LL": ll.copy(), "LH": lh.copy(), "HL": hl.copy(), "HH": hh.copy()}
        positions = self._select_positions(band_map, len(bits), content_id)
        signs = self._signs(len(positions), content_id)

        k = 0
        for bit in bits:
            for _ in range(self.repeats):
                band_name, by, bx = positions[k]
                band = band_map[band_name]
                y0 = by * self.block_size
                x0 = bx * self.block_size
                block = np.array(band[y0:y0 + self.block_size, x0:x0 + self.block_size], dtype=np.float32)
                dct_block = cv2.dct(block)
                u, v = self.coeff_pos
                signed_coeff = signs[k] * dct_block[u, v]
                dct_block[u, v] = signs[k] * self._qim_embed(float(signed_coeff), int(bit), self.delta)
                band[y0:y0 + self.block_size, x0:x0 + self.block_size] = cv2.idct(dct_block)
                k += 1

        watermarked = self._merge_bands(ycrcb, band_map["LL"], band_map["LH"], band_map["HL"], band_map["HH"])
        return EmbedResult(image=watermarked, logical_bits=bits)

    def extract(self, image_bgr: np.ndarray, n_bits: int, content_id: str) -> np.ndarray:
        _, ll, lh, hl, hh = self._split_bands(image_bgr)
        band_map = {"LL": ll, "LH": lh, "HL": hl, "HH": hh}
        positions = self._select_positions(band_map, n_bits, content_id)
        signs = self._signs(len(positions), content_id)

        hard_bits = np.zeros((n_bits, self.repeats), dtype=np.uint8)
        conf = np.zeros((n_bits, self.repeats), dtype=np.float32)
        k = 0
        for i in range(n_bits):
            for r in range(self.repeats):
                band_name, by, bx = positions[k]
                band = band_map[band_name]
                y0 = by * self.block_size
                x0 = bx * self.block_size
                block = np.array(band[y0:y0 + self.block_size, x0:x0 + self.block_size], dtype=np.float32)
                dct_block = cv2.dct(block)
                u, v = self.coeff_pos
                signed_coeff = signs[k] * dct_block[u, v]
                bit, c = self._qim_detect(float(signed_coeff), self.delta)
                hard_bits[i, r] = bit
                conf[i, r] = c
                k += 1

        sums = hard_bits.sum(axis=1)
        majority = (sums > (self.repeats / 2.0)).astype(np.uint8)
        ties = sums == (self.repeats / 2.0)
        if np.any(ties):
            majority[ties] = (conf[ties].sum(axis=1) > 0).astype(np.uint8)
        return majority

    @staticmethod
    def bit_error_rate(reference_bits: np.ndarray, estimated_bits: np.ndarray) -> float:
        ref = np.asarray(reference_bits, dtype=np.uint8).flatten()
        est = np.asarray(estimated_bits, dtype=np.uint8).flatten()
        if ref.shape != est.shape:
            raise ValueError("bit arrays must have the same shape")
        return float(np.mean(ref != est))

    @staticmethod
    def bits_to_noise_image(bits: np.ndarray, scale: int = 8) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        side = int(np.ceil(np.sqrt(len(bits))))
        canvas = np.zeros(side * side, dtype=np.uint8)
        canvas[: len(bits)] = bits * 255
        img = canvas.reshape(side, side)
        return cv2.resize(img, (side * scale, side * scale), interpolation=cv2.INTER_NEAREST)


def average_collusion(images: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    return np.clip(np.mean(stack, axis=0), 0, 255).astype(np.uint8)


def xor_collusion_via_reembed(base_image: np.ndarray, colluder_images: list[np.ndarray], wm: BlindDwtDctQim, n_bits: int, content_id: str) -> tuple[np.ndarray, np.ndarray]:
    extracted = [wm.extract(img, n_bits=n_bits, content_id=content_id) for img in colluder_images]
    fused = np.bitwise_xor.reduce(np.stack(extracted, axis=0), axis=0).astype(np.uint8)
    pirate = wm.embed(base_image, fused, content_id=content_id).image
    return pirate, fused
