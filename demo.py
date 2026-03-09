from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tardos import UserRecord, accuse_symmetric, build_codebook, rank_suspects, sample_tardos_probabilities
from watermark import BlindDwtDctQim, average_collusion, xor_collusion_via_reembed

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


def save_rgb(path: Path, bgr: np.ndarray) -> None:
    cv2.imwrite(str(path), bgr)


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

    parser.add_argument("--delta", type=float, default=26.0,
                        help="QIM量化步长")
    parser.add_argument("--repeats", type=int, default=6,
                        help="每个bit重复嵌入次数")

    parser.add_argument("--master-key", type=str, default="server-master-key-2026",
                        help="系统主密钥")
    parser.add_argument("--content-id", type=str, default="asset-demo-001",
                        help="内容ID")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Tardos概率向量随机种子")

    parser.add_argument("--cover-size", type=int, default=512,
                        help="未提供图片时，自动生成合成图的尺寸")

    return parser.parse_args()


def load_cover_image(image_path: str | None, cover_size: int) -> np.ndarray:
    if image_path is None:
        return make_synthetic_cover(cover_size)

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return img


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

    cover = load_cover_image(args.image, args.cover_size)
    save_rgb(out / "cover.png", cover)

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
        bands=("LL", "LH"),
    )

    watermarked = {}
    clean_extract_ber = {}
    psnr_map = {}

    for idx, user in enumerate(users):
        img = wm.embed(cover, codebook[idx], content_id=args.content_id).image
        watermarked[user.user_id] = img

        extracted = wm.extract(img, n_bits=args.code_length, content_id=args.content_id)
        clean_extract_ber[user.user_id] = wm.bit_error_rate(codebook[idx], extracted)

        mse = float(np.mean((cover.astype(np.float32) - img.astype(np.float32)) ** 2))
        psnr_map[user.user_id] = 99.0 if mse <= 1e-12 else float(
            10.0 * np.log10((255.0 ** 2) / mse)
        )

        save_rgb(out / f"{user.user_id}_watermarked.png", img)
        save_rgb(out / f"{user.user_id}_extracted_bits.png", wm.bits_to_noise_image(extracted))

    colluder_imgs = [watermarked[cid] for cid in colluder_ids]

    pirate_avg = average_collusion(colluder_imgs)
    avg_bits = wm.extract(pirate_avg, n_bits=args.code_length, content_id=args.content_id)
    avg_scores = accuse_symmetric(avg_bits, codebook, p)
    avg_ranking = rank_suspects(users, avg_scores)
    save_rgb(out / "pirate_average.png", pirate_avg)
    save_rgb(out / "pirate_average_bits.png", wm.bits_to_noise_image(avg_bits))

    pirate_xor, xor_bits_fused = xor_collusion_via_reembed(
        base_image=pirate_avg,
        colluder_images=colluder_imgs,
        wm=wm,
        n_bits=args.code_length,
        content_id=args.content_id,
    )
    xor_bits = wm.extract(pirate_xor, n_bits=args.code_length, content_id=args.content_id)
    xor_scores = accuse_symmetric(xor_bits, codebook, p)
    xor_ranking = rank_suspects(users, xor_scores)
    save_rgb(out / "pirate_xor.png", pirate_xor)
    save_rgb(out / "pirate_xor_bits.png", wm.bits_to_noise_image(xor_bits))

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
            "colluders": colluder_ids,
            "delta": args.delta,
            "repeats": args.repeats,
            "content_id": args.content_id,
            "master_key": args.master_key,
            "seed": args.seed,
        },
        "clean_ber": clean_extract_ber,
        "psnr": psnr_map,
        "average_attack": {
            "top5": avg_ranking[:5],
            "top3_hit_count": sum(uid in colluder_ids for uid, _ in avg_ranking[:3]),
        },
        "xor_attack": {
            "top5": xor_ranking[:5],
            "top3_hit_count": sum(uid in colluder_ids for uid, _ in xor_ranking[:3]),
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