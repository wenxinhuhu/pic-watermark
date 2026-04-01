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


@dataclass(frozen=True)
class TileJob:
    tile_index: int
    y0: int
    y1: int
    x0: int
    x1: int
    bit_offset: int
    n_bits: int
    tile_content_id: str


class BlindDwtDctQim:
    """Blind watermarking with DWT + block-DCT + scalar QIM.

    - Blind extraction: original image is not required.
    - Keyed placement: selected blocks and coefficient signs depend on master_key + content_id.
    - Repetition code: each logical bit is embedded in multiple blocks and decoded by majority vote.

    新增：支持分块规划。这样大图可以按 tile 逐块处理，而不是一次性把整张图读进内存。
    分块模式下，每个 tile 独立做 DWT-DCT-QIM；提取时只要 tile 划分和 content_id 一致，就能盲提取。
    """

    def __init__(
        self,
        master_key: str,
        wavelet: str = "haar",
        block_size: int = 8,
        coeff_pos: tuple[int, int] = (3, 3),
        delta: float = 8.0,
        repeats: int = 2,
        bands: tuple[str, ...] = ("LH", "HL"),
        delta_mode: str = "normalized_8bit",
    ) -> None:
        self.master_key = master_key
        self.wavelet = wavelet
        self.block_size = block_size
        self.coeff_pos = coeff_pos
        self.delta = float(delta)
        self.repeats = int(repeats)
        self.bands = bands
        self.delta_mode = delta_mode
        self._tile_capacity_cache: dict[tuple[int, int], int] = {}

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

    def _working_delta(self, src_dtype: np.dtype) -> float:
        peak = self._pixel_peak(src_dtype)

        if self.delta_mode == "normalized_8bit":
            # delta 按 8-bit 码值理解，视觉强度在不同位深下更接近
            return self.delta / 255.0

        if self.delta_mode == "native":
            # delta 直接按原图 native code value 理解
            return self.delta / peak

        raise ValueError(f"未知 delta_mode: {self.delta_mode}")

    @staticmethod
    def _pixel_peak(dtype: np.dtype) -> float:
        dtype = np.dtype(dtype)
        if dtype == np.uint8:
            return 255.0
        if dtype == np.uint16:
            return 65535.0
        raise ValueError(f"不支持的图像 dtype: {dtype}")

    @classmethod
    def _to_working_float(cls, image_bgr: np.ndarray) -> tuple[np.ndarray, np.dtype, float]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"只支持 3 通道 BGR 图像，当前 shape={image_bgr.shape}")

        src_dtype = image_bgr.dtype
        peak = cls._pixel_peak(src_dtype)

        image_f32 = image_bgr.astype(np.float32) / peak
        return image_f32, src_dtype, peak

    @classmethod
    def _from_working_float(cls, image_bgr_f32: np.ndarray, dst_dtype: np.dtype) -> np.ndarray:
        peak = cls._pixel_peak(dst_dtype)
        out = np.clip(image_bgr_f32, 0.0, 1.0)
        out = np.rint(out * peak)

        if np.dtype(dst_dtype) == np.uint8:
            return out.astype(np.uint8)
        if np.dtype(dst_dtype) == np.uint16:
            return out.astype(np.uint16)

        raise ValueError(f"不支持的输出 dtype: {dst_dtype}")

    def _split_bands(self, image_bgr: np.ndarray):
        image_f32, src_dtype, peak = self._to_working_float(image_bgr)

        ycrcb = cv2.cvtColor(image_f32, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]

        coeffs = pywt.dwt2(y, self.wavelet)
        ll, (lh, hl, hh) = coeffs
        return ycrcb, ll, lh, hl, hh, src_dtype, peak

    def _merge_bands(self, ycrcb: np.ndarray, ll, lh, hl, hh, dst_dtype: np.dtype) -> np.ndarray:
        y_rec = pywt.idwt2((ll, (lh, hl, hh)), self.wavelet)
        h, w = ycrcb.shape[:2]
        y_rec = y_rec[:h, :w]

        y_rec = np.clip(y_rec, 0.0, 1.0)

        out = ycrcb.copy()
        out[:, :, 0] = y_rec

        bgr_f32 = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
        bgr_f32 = np.clip(bgr_f32, 0.0, 1.0)

        return self._from_working_float(bgr_f32, dst_dtype)

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
        ycrcb, ll, lh, hl, hh, src_dtype, _ = self._split_bands(image_bgr)
        delta_work = self._working_delta(src_dtype)

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
                block = np.array(
                    band[y0:y0 + self.block_size, x0:x0 + self.block_size],
                    dtype=np.float32,
                )
                dct_block = cv2.dct(block)
                u, v = self.coeff_pos
                signed_coeff = signs[k] * dct_block[u, v]
                dct_block[u, v] = signs[k] * self._qim_embed(
                    float(signed_coeff), int(bit), delta_work
                )
                band[y0:y0 + self.block_size, x0:x0 + self.block_size] = cv2.idct(dct_block)
                k += 1

        watermarked = self._merge_bands(
            ycrcb,
            band_map["LL"],
            band_map["LH"],
            band_map["HL"],
            band_map["HH"],
            dst_dtype=src_dtype,
        )
        return EmbedResult(image=watermarked, logical_bits=bits)

    def extract(self, image_bgr: np.ndarray, n_bits: int, content_id: str) -> np.ndarray:
        delta_work = self._working_delta(src_dtype=image_bgr.dtype)
        _, ll, lh, hl, hh, _, _ = self._split_bands(image_bgr)
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
                bit, c = self._qim_detect(float(signed_coeff), delta_work)
                hard_bits[i, r] = bit
                conf[i, r] = c
                k += 1

        sums = hard_bits.sum(axis=1)
        majority = (sums > (self.repeats / 2.0)).astype(np.uint8)
        ties = sums == (self.repeats / 2.0)
        if np.any(ties):
            majority[ties] = (conf[ties].sum(axis=1) > 0).astype(np.uint8)
        return majority

    def tile_capacity(self, tile_h: int, tile_w: int) -> int:
        key = (int(tile_h), int(tile_w))
        if key in self._tile_capacity_cache:
            return self._tile_capacity_cache[key]

        if tile_h < 2 * self.block_size or tile_w < 2 * self.block_size:
            self._tile_capacity_cache[key] = 0
            return 0

        dummy = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        _, ll, lh, hl, hh, _, _ = self._split_bands(dummy)
        band_map = {"LL": ll, "LH": lh, "HL": hl, "HH": hh}
        capacity = len(self._candidate_positions(band_map)) // max(self.repeats, 1)
        self._tile_capacity_cache[key] = int(capacity)
        return int(capacity)

    def build_tiled_jobs(
        self,
        image_shape: tuple[int, int, int] | tuple[int, int],
        n_bits: int,
        content_id: str,
        tile_size: int = 2048,
    ) -> list[TileJob]:
        if tile_size < 2 * self.block_size:
            raise ValueError(f"tile_size 太小，至少需要 >= {2 * self.block_size}")

        h = int(image_shape[0])
        w = int(image_shape[1])
        bit_offset = 0
        jobs: list[TileJob] = []
        tile_index = 0

        for y0 in range(0, h, tile_size):
            y1 = min(y0 + tile_size, h)
            for x0 in range(0, w, tile_size):
                x1 = min(x0 + tile_size, w)
                capacity = self.tile_capacity(y1 - y0, x1 - x0)
                if capacity <= 0:
                    tile_index += 1
                    continue

                take = min(capacity, n_bits - bit_offset)
                if take > 0:
                    jobs.append(
                        TileJob(
                            tile_index=tile_index,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                            bit_offset=bit_offset,
                            n_bits=take,
                            tile_content_id=f"{content_id}::tile::{tile_index}::{y0}_{x0}",
                        )
                    )
                    bit_offset += take
                    if bit_offset >= n_bits:
                        return jobs
                tile_index += 1

        if bit_offset < n_bits:
            raise ValueError(
                f"大图分块容量不足：需要 {n_bits} bits，当前 tile_size={tile_size} 仅能放下 {bit_offset} bits"
            )
        return jobs

    def embed_tiled_rgb_inplace(
        self,
        image_rgb: np.ndarray,
        bits: np.ndarray,
        content_id: str,
        tile_size: int = 2048,
    ) -> list[TileJob]:
        bits = np.asarray(bits, dtype=np.uint8).flatten()
        jobs = self.build_tiled_jobs(image_rgb.shape, len(bits), content_id, tile_size=tile_size)

        for job in jobs:
            tile_rgb = np.asarray(image_rgb[job.y0:job.y1, job.x0:job.x1, :])
            tile_bgr = cv2.cvtColor(np.array(tile_rgb, copy=True), cv2.COLOR_RGB2BGR)
            embedded = self.embed(
                tile_bgr,
                bits[job.bit_offset:job.bit_offset + job.n_bits],
                content_id=job.tile_content_id,
            ).image
            image_rgb[job.y0:job.y1, job.x0:job.x1, :] = cv2.cvtColor(embedded, cv2.COLOR_BGR2RGB)

        return jobs

    def extract_tiled_rgb(
        self,
        image_rgb: np.ndarray,
        n_bits: int,
        content_id: str,
        tile_size: int = 2048,
    ) -> np.ndarray:
        jobs = self.build_tiled_jobs(image_rgb.shape, n_bits, content_id, tile_size=tile_size)
        out = np.zeros(n_bits, dtype=np.uint8)

        for job in jobs:
            tile_rgb = np.asarray(image_rgb[job.y0:job.y1, job.x0:job.x1, :])
            tile_bgr = cv2.cvtColor(np.array(tile_rgb, copy=False), cv2.COLOR_RGB2BGR)
            bits_part = self.extract(tile_bgr, n_bits=job.n_bits, content_id=job.tile_content_id)
            out[job.bit_offset:job.bit_offset + job.n_bits] = bits_part

        return out

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
    if not images:
        raise ValueError("images 不能为空")

    dtype = images[0].dtype
    peak = BlindDwtDctQim._pixel_peak(dtype)

    for img in images:
        if img.dtype != dtype:
            raise ValueError("所有合谋图像的 dtype 必须一致")
        if img.shape != images[0].shape:
            raise ValueError("所有合谋图像的 shape 必须一致")

    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    avg = np.mean(stack, axis=0)
    avg = np.clip(np.rint(avg), 0, peak)
    return avg.astype(dtype)


def xor_collusion_via_reembed(base_image: np.ndarray, colluder_images: list[np.ndarray], wm: BlindDwtDctQim, n_bits: int, content_id: str) -> tuple[np.ndarray, np.ndarray]:
    extracted = [wm.extract(img, n_bits=n_bits, content_id=content_id) for img in colluder_images]
    fused = np.bitwise_xor.reduce(np.stack(extracted, axis=0), axis=0).astype(np.uint8)
    pirate = wm.embed(base_image, fused, content_id=content_id).image
    return pirate, fused
