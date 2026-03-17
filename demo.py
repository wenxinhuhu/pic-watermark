from __future__ import annotations

import argparse
import json
from pathlib import Path

import zlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

from tardos import UserRecord, accuse_symmetric, build_codebook, rank_suspects, sample_tardos_probabilities
from watermark import BlindDwtDctQim, average_collusion, xor_collusion_via_reembed
from psb_bridge import (
    SourceMeta,
    read_cover_asset,
    save_bgr_preview_png,
    save_bgr_as_flattened_psb_via_photoshop,
    save_bgr_native_tiff,
)

DISPLAY_TEXT_BYTES = 40  #可读字符串最多 40 字节，够放 UID/CID

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    pad = (-len(bits)) % 8
    if pad:
        bits = np.pad(bits, (0, pad), constant_values=0)
    return np.packbits(bits).tobytes()

def build_display_text(user_id: str, content_id: str) -> str:
    return f"UID={user_id};CID={content_id}"

def encode_display_text(text: str, max_text_bytes: int = DISPLAY_TEXT_BYTES) -> np.ndarray:
    raw = text.encode("utf-8")
    if len(raw) > max_text_bytes:
        raise ValueError(f"展示字符串过长，最多 {max_text_bytes} 字节，当前 {len(raw)}")
    body = raw.ljust(max_text_bytes, b"\0")
    crc = zlib.crc32(raw).to_bytes(4, "big")
    packet = len(raw).to_bytes(2, "big") + body + crc
    return bytes_to_bits(packet)

def decode_display_text(bits: np.ndarray, max_text_bytes: int = DISPLAY_TEXT_BYTES) -> tuple[str, bool]:
    data = bits_to_bytes(bits)
    packet_len = 2 + max_text_bytes + 4
    if len(data) < packet_len:
        return "", False

    data = data[:packet_len]
    n = int.from_bytes(data[:2], "big")
    if n > max_text_bytes:
        return "", False

    raw = data[2:2+n]
    crc_saved = data[2 + max_text_bytes: 2 + max_text_bytes + 4]
    crc_now = zlib.crc32(raw).to_bytes(4, "big")
    ok = (crc_saved == crc_now)

    try:
        text = raw.decode("utf-8")
    except Exception:
        return "", False

    return text, ok

DISPLAY_BITS_LEN = (2 + DISPLAY_TEXT_BYTES + 4) * 8

USER_DB = [
    UserRecord(user_id="user_1",  user_key="k_9f2A1x7P"),
    UserRecord(user_id="user_2",  user_key="k_B8mQ3d2L"),
    UserRecord(user_id="user_3",  user_key="k_T4zN9v1R"),
    UserRecord(user_id="user_4",  user_key="k_H7pX5s8W"),
    UserRecord(user_id="user_5",  user_key="k_M2cJ6u4K"),
    UserRecord(user_id="user_6",  user_key="k_R5yD8n3F"),
    UserRecord(user_id="user_7",  user_key="k_V1qL7b9T"),
    UserRecord(user_id="user_8",  user_key="k_P6wE2m5C"),
    UserRecord(user_id="user_9",  user_key="k_N3rK4x8Z"),
    UserRecord(user_id="user_10", user_key="k_G8tU1p6Y"),
]

def make_synthetic_cover(size: int = 512) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 110 + 40 * np.sin(x / 17.0) + 35 * np.cos(y / 29.0)
    grad = 0.25 * x + 0.15 * y
    tex = 12 * np.sin((x + y) / 11.0) + 8 * np.cos((x - y) / 19.0)
    b = np.clip(base + grad * 0.15 + tex, 0, 255)
    g = np.clip(base + grad * 0.10 - tex * 0.8 + 20, 0, 255)
    r = np.clip(base + grad * 0.12 + tex * 0.5 + 35, 0, 255)
    img = np.stack([b, g, r], axis=-1).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 5, (220, 160, 90), thickness=-1)
    cv2.rectangle(img, (40, 60), (190, 170), (80, 200, 230), thickness=-1)
    cv2.putText(img, "WM DEMO", (size // 5, size - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (245, 245, 245), 3, cv2.LINE_AA)
    return img

def dtype_peak(dtype: np.dtype) -> float:
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 255.0
    if dtype == np.uint16:
        return 65535.0
    raise ValueError(f"不支持的 dtype: {dtype}")


def to_preview_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    if image.dtype == np.uint16:
        preview = np.rint(image.astype(np.float32) / 65535.0 * 255.0)
        return np.clip(preview, 0, 255).astype(np.uint8)

    raise ValueError(f"不支持预览转换的 dtype: {image.dtype}")

def save_rgb(path: Path, bgr: np.ndarray) -> None:
    cv2.imwrite(str(path), to_preview_u8(bgr))


def plot_scores(path: Path, score_dict: dict[str, list[tuple[str, float]]], colluders: set[str]) -> None:
    fig, axes = plt.subplots(1, len(score_dict), figsize=(6 * len(score_dict), 5))
    if len(score_dict) == 1:
        axes = [axes]

    for ax, (title, ranking) in zip(axes, score_dict.items()):
        users = [u for u, _ in ranking]
        scores = [s for _, s in ranking]
        colors = ["tab:red" if u in colluders else "tab:blue" for u in users]
        ax.bar(users, scores, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Tardos score")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Blind watermark demo with Tardos + DWT-DCT + QIM")

    parser.add_argument("--image", type=str, default=None,
                        help="待嵌入水印的图片路径；不传则自动生成合成图")
    parser.add_argument("--out-dir", type=str, default="results",
                        help="输出目录")

    parser.add_argument("--num-users", type=int, default=8,
                        help="用户数量")
    parser.add_argument("--code-length", type=int, default=320,
                        help="Tardos码长")
    parser.add_argument("--colluders", type=str, default="user_2,user_5,user_7",
                        help="合谋用户，逗号分隔，例如 user_2,user_5,user_7")

    parser.add_argument("--delta", type=float, default=8.0,
                        help="QIM量化步长")
    parser.add_argument("--repeats", type=int, default=2,
                        help="每个bit重复嵌入次数")
    parser.add_argument("--delta-mode", type=str, default="normalized_8bit",
                    choices=["normalized_8bit", "native"],
                    help="delta 的解释方式")
    
    parser.add_argument("--master-key", type=str, default="server-master-key-2026",
                        help="系统主密钥")
    parser.add_argument("--content-id", type=str, default="asset-demo-001",
                        help="内容ID")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Tardos概率向量随机种子")

    parser.add_argument("--cover-size", type=int, default=512,
                        help="未提供图片时，自动生成合成图的尺寸")
    parser.add_argument("--emit-psb", action="store_true",
                        help="如果输入是 PSD/PSB，则额外输出扁平单层 PSB 分发文件")

    parser.add_argument("--psb-max-side", type=int, default=0,
                        help="读取 PSD/PSB 时可选缩边；0 表示不缩放，直接使用原始尺寸（可能导致内存占用过大）")
    return parser.parse_args()


# def load_cover_image(image_path: str | None, cover_size: int) -> np.ndarray:
#     if image_path is None:
#         return make_synthetic_cover(cover_size)

#     img = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise FileNotFoundError(f"无法读取图片: {image_path}")
#     return img

def load_cover_and_meta(
    image_path: str | None,
    cover_size: int,
    psb_max_side: int,
) -> tuple[np.ndarray, SourceMeta]:
    if image_path is None:
        img = make_synthetic_cover(cover_size)
        meta = SourceMeta(
            input_path=None,
            input_ext=".png",
            input_is_psb=False,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            depth=8,
        )
        return img, meta

    return read_cover_asset(image_path, max_side=psb_max_side)

def save_main_asset(
    out_dir: Path,
    stem: str,
    bgr: np.ndarray,
    source_meta: SourceMeta,
    emit_psb: bool,
) -> dict:
    result = {
        "png": None,
        "tif": None,
        "psb": None,
        "psb_error": None,
    }
    # 1) 正式保真输出：TIFF
    tif_path = out_dir / f"{stem}.tif"
    save_bgr_native_tiff(tif_path, bgr, source_meta)
    result["tif"] = str(tif_path)

    # 2) 网页/演示预览：PNG
    png_path = out_dir / f"{stem}.png"
    save_bgr_preview_png(png_path, bgr)
    result["png"] = str(png_path)

    # 3) PSB 分发输出
    if emit_psb:
        psb_path = out_dir / f"{stem}.psb"
        try:
            save_bgr_as_flattened_psb_via_photoshop(psb_path, bgr, dpi_x=source_meta.dpi_x, dpi_y=source_meta.dpi_y, icc_profile=source_meta.icc_profile)
            result["psb"] = str(psb_path)
        except Exception as exc:
            result["psb_error"] = str(exc)

    return result

def build_users(num_users: int) -> list[UserRecord]:
    if num_users < 1:
        raise ValueError("num_users 必须 >= 1")
    if num_users > len(USER_DB):
        raise ValueError(f"当前代码里只维护了 {len(USER_DB)} 个用户，请把 num_users 设为 <= {len(USER_DB)}")
    return USER_DB[:num_users]


def parse_colluders(colluders_str: str, valid_user_ids: set[str]) -> list[str]:
    colluders = [x.strip() for x in colluders_str.split(",") if x.strip()]
    if not colluders:
        raise ValueError("colluders 不能为空")

    invalid = [x for x in colluders if x not in valid_user_ids]
    if invalid:
        raise ValueError(f"非法合谋用户ID: {invalid}，可选用户: {sorted(valid_user_ids)}")
    return colluders


def run_demo(args) -> dict:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # cover = load_cover_image(args.image, args.cover_size)
    # save_rgb(out / "cover.png", cover)

    cover, source_meta = load_cover_and_meta(
        args.image,
        args.cover_size,
        args.psb_max_side,
    )
    if source_meta.input_is_psb and source_meta.depth == 16 and cover.dtype != np.uint16:
        raise RuntimeError(
            f"检测到输入是 16-bit PSB，但当前读入数组 dtype={cover.dtype}。"
            "这说明输入桥没有保住 16-bit，当前结果不能视为保真输出。"
        )
    if source_meta.input_is_psb and args.psb_max_side != 0:
        raise RuntimeError("保真模式下，PSB 输入时 psb_max_side 必须为 0；否则会发生缩图。")
    cover_paths = save_main_asset(out, "cover", cover, source_meta, emit_psb=args.emit_psb)

    users = build_users(args.num_users)
    user_ids = {u.user_id for u in users}
    colluder_ids = parse_colluders(args.colluders, user_ids)

    p = sample_tardos_probabilities(
        length=args.code_length,
        cutoff=0.08,
        seed=args.seed,
    )
    codebook = build_codebook(users, p, master_key=args.master_key)

    wm = BlindDwtDctQim(
        master_key=args.master_key,
        delta=args.delta,
        repeats=args.repeats,
        coeff_pos=(3, 3),
        bands=("LH", "HL"),#LL感觉变动容易引起色差
        delta_mode=args.delta_mode,
    )

    watermarked = {}
    clean_extract_ber = {}
    psnr_map = {}
    user_outputs = {}
    user_results = {}

    for idx, user in enumerate(users):
        display_text = build_display_text(user.user_id, args.content_id)
        display_bits = encode_display_text(display_text)
        fingerprint_bits = codebook[idx]

        payload_bits = np.concatenate([display_bits, fingerprint_bits]).astype(np.uint8)

        img = wm.embed(cover, payload_bits, content_id=args.content_id).image
        watermarked[user.user_id] = img

        extracted = wm.extract(img, n_bits=len(payload_bits), content_id=args.content_id)

        extracted_display_bits = extracted[:DISPLAY_BITS_LEN]
        extracted_fingerprint_bits = extracted[DISPLAY_BITS_LEN:]

        extracted_text, text_ok = decode_display_text(extracted_display_bits)
        clean_extract_ber[user.user_id] = wm.bit_error_rate(
            fingerprint_bits, extracted_fingerprint_bits
        )

        peak = dtype_peak(cover.dtype)
        mse = float(np.mean((cover.astype(np.float32) - img.astype(np.float32)) ** 2))
        psnr_map[user.user_id] = 99.0 if mse <= 1e-12 else float(
            10.0 * np.log10((peak ** 2) / mse)
        )

        user_paths = save_main_asset(
            out, f"{user.user_id}_watermarked", img, source_meta, emit_psb=args.emit_psb
        )
        user_outputs[user.user_id] = user_paths

        user_results[user.user_id] = {
            "embedded_text": display_text,
            "extracted_text": extracted_text,
            "text_ok": text_ok,
            "fingerprint_ber": clean_extract_ber[user.user_id],
        }

        # 这里只保存“指纹部分”的噪声图，不保存整段 payload
        save_rgb(
            out / f"{user.user_id}_extracted_bits.png",
            wm.bits_to_noise_image(extracted_fingerprint_bits)
        )


    colluder_imgs = [watermarked[cid] for cid in colluder_ids]

    pirate_avg = average_collusion(colluder_imgs)
    total_bits = DISPLAY_BITS_LEN + args.code_length

    avg_bits = wm.extract(pirate_avg, n_bits=total_bits, content_id=args.content_id)
    avg_text, avg_text_ok = decode_display_text(avg_bits[:DISPLAY_BITS_LEN])
    avg_fp_bits = avg_bits[DISPLAY_BITS_LEN:]
    avg_scores = accuse_symmetric(avg_fp_bits, codebook, p)
    avg_ranking = rank_suspects(users, avg_scores)

    pirate_avg_paths = save_main_asset(
        out, "pirate_average", pirate_avg, source_meta, emit_psb=args.emit_psb
    )
    save_rgb(out / "pirate_average_bits.png", wm.bits_to_noise_image(avg_fp_bits))

    pirate_xor, xor_bits_fused = xor_collusion_via_reembed(
        base_image=pirate_avg,
        colluder_images=colluder_imgs,
        wm=wm,
        n_bits=total_bits,
        content_id=args.content_id,
    )

    xor_bits = wm.extract(pirate_xor, n_bits=total_bits, content_id=args.content_id)
    xor_text, xor_text_ok = decode_display_text(xor_bits[:DISPLAY_BITS_LEN])
    xor_fp_bits = xor_bits[DISPLAY_BITS_LEN:]
    xor_scores = accuse_symmetric(xor_fp_bits, codebook, p)
    xor_ranking = rank_suspects(users, xor_scores)

    pirate_xor_paths = save_main_asset(
        out, "pirate_xor", pirate_xor, source_meta, emit_psb=args.emit_psb
    )
    save_rgb(out / "pirate_xor_bits.png", wm.bits_to_noise_image(xor_fp_bits))


    plot_scores(
        out / "scores.png",
        {
            "Average collusion": avg_ranking,
            "XOR-like collusion": xor_ranking,
        },
        colluders=set(colluder_ids),
    )

    diff = cv2.absdiff(cover, watermarked[users[0].user_id])
    save_rgb(out / "diff_first_user.png", diff)

    report = {
        "config": {
            "image": args.image if args.image is not None else "synthetic",
            "num_users": args.num_users,
            "code_length": args.code_length,
            "display_bits_len": DISPLAY_BITS_LEN,
            "total_bits": DISPLAY_BITS_LEN + args.code_length,
            "colluders": colluder_ids,
            "delta": args.delta,
            "repeats": args.repeats,
            "content_id": args.content_id,
            "master_key": args.master_key,
            "seed": args.seed,
            "emit_psb": args.emit_psb,
            "psb_max_side": args.psb_max_side,
            "input_is_psb": source_meta.input_is_psb,
            "input_depth": source_meta.depth,
            "delta_mode": args.delta_mode,
        },
        "clean_ber": clean_extract_ber,
        "psnr": psnr_map,
        "user_results": user_results,
        "average_attack": {
            "decoded_text": avg_text,
            "decoded_text_ok": avg_text_ok,
            "top5": avg_ranking[:5],
            "top3_hit_count": sum(uid in colluder_ids for uid, _ in avg_ranking[:3]),
        },
        "xor_attack": {
            "decoded_text": xor_text,
            "decoded_text_ok": xor_text_ok,
            "top5": xor_ranking[:5],
            "top3_hit_count": sum(uid in colluder_ids for uid, _ in xor_ranking[:3]),
        },
        "outputs": {
            "cover": cover_paths,
            "users": user_outputs,
            "pirate_average": pirate_avg_paths,
            "pirate_xor": pirate_xor_paths,
        },
    }

    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    args = parse_args()
    report = run_demo(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))