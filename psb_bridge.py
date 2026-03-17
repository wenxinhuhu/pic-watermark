from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage


@dataclass
class SourceMeta:
    input_path: str | None
    input_ext: str
    input_is_psb: bool
    width: int
    height: int
    depth: int | None = None


def _bgr_to_rgb_pil(bgr: np.ndarray) -> Image.Image:
    if bgr is None:
        raise ValueError("bgr 不能为空")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"只支持 3 通道图像，当前 shape={bgr.shape}")
    if bgr.dtype != np.uint8:
        raise ValueError(f"当前仅支持 uint8 输出，当前 dtype={bgr.dtype}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def read_cover_asset(image_path: str, max_side: int = 0) -> tuple[np.ndarray, SourceMeta]:
    path = Path(image_path)
    ext = path.suffix.lower()

    if ext not in {".psd", ".psb"}:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        meta = SourceMeta(
            input_path=str(path),
            input_ext=ext,
            input_is_psb=False,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            depth=8,
        )
        return img, meta

    psd = PSDImage.open(path)
    pil_img = psd.composite().convert("RGB")

    if max_side and max_side > 0:
        w, h = pil_img.size
        if max(w, h) > max_side:
            pil_img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    rgb = np.asarray(pil_img, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    meta = SourceMeta(
        input_path=str(path),
        input_ext=ext,
        input_is_psb=True,
        width=int(getattr(psd, "width", bgr.shape[1])),
        height=int(getattr(psd, "height", bgr.shape[0])),
        depth=getattr(psd, "depth", None),
    )
    return bgr, meta


def save_bgr_preview_png(path: str | Path, bgr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"写入 PNG 失败: {path}")
    return path


def _dispatch_photoshop():
    """
    尝试连接 Photoshop COM；失败时抛出更清晰的错误。
    """
    try:
        from win32com.client import Dispatch
        return Dispatch("Photoshop.Application")
    except Exception as exc:
        raise RuntimeError(
            "无法连接 Photoshop COM（Photoshop.Application）。"
            "通常是因为本机未安装 Photoshop，或 Photoshop 的 COM 注册不可用。"
            "当前可以先关闭 --emit-psb，仅输出 PNG；"
            "若一定要输出 PSB，请确认 Photoshop 已安装并可被 COM 调用。"
        ) from exc


def save_bgr_as_flattened_psb_via_photoshop(path: str | Path, bgr: np.ndarray) -> Path:
    from win32com.client import Dispatch

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pil_img = _bgr_to_rgb_pil(bgr)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_png = Path(tmpdir) / "wm_tmp.png"
        pil_img.save(tmp_png, format="PNG", compress_level=0)

        app = _dispatch_photoshop()
        app.DisplayDialogs = 3

        doc = app.Open(str(tmp_png.resolve()))
        try:
            jsx = rf"""
#target photoshop
(function () {{
    var theFile = new File("{str(out_path.resolve()).replace("\\", "\\\\")}");
    var s2t = function (s) {{ return app.stringIDToTypeID(s); }};
    var descriptor = new ActionDescriptor();
    var descriptor2 = new ActionDescriptor();
    descriptor.putObject(s2t("as"), s2t("largeDocumentFormat"), descriptor2);
    descriptor.putPath(s2t("in"), theFile);
    descriptor.putBoolean(s2t("copy"), true);
    descriptor.putBoolean(s2t("lowerCase"), true);
    executeAction(s2t("save"), descriptor, DialogModes.NO);
}}());
"""
            app.DoJavaScript(jsx)
        finally:
            doc.Close(2)

    if not out_path.exists():
        raise RuntimeError(f"Photoshop 未成功生成 PSB: {out_path}")

    return out_path