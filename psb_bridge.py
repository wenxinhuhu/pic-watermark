from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
import tifffile

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
    dpi_x: float | None = None
    dpi_y: float | None = None
    icc_profile: bytes | None = None

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

def read_image_meta(path: str | Path) -> tuple[float | None, float | None, bytes | None]:
    path = Path(path)
    dpi_x = None
    dpi_y = None
    icc_profile = None

    try:
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
            if dpi and len(dpi) >= 2:
                dpi_x, dpi_y = float(dpi[0]), float(dpi[1])

            resolution = im.info.get("resolution")
            if (dpi_x is None or dpi_y is None) and resolution and len(resolution) >= 2:
                dpi_x, dpi_y = float(resolution[0]), float(resolution[1])

            icc_profile = im.info.get("icc_profile")
    except Exception:
        pass

    return dpi_x, dpi_y, icc_profile

def write_temp_raster_for_photoshop(
    tmpdir: str | Path,
    bgr: np.ndarray,
    dpi_x: float | None = None,
    dpi_y: float | None = None,
    icc_profile: bytes | None = None,
) -> Path:
    tmpdir = Path(tmpdir)

    if bgr.dtype == np.uint8:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tmp_png = tmpdir / "wm_tmp.png"

        save_kwargs = {
            "format": "PNG",
            "compress_level": 0,
        }
        if dpi_x and dpi_y:
            save_kwargs["dpi"] = (dpi_x, dpi_y)
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        Image.fromarray(rgb).save(tmp_png, **save_kwargs)
        return tmp_png

    if bgr.dtype == np.uint16:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tmp_tif = tmpdir / "wm_tmp.tif"

        save_kwargs = {
            "photometric": "rgb",
            "compression": None,
        }
        if dpi_x and dpi_y:
            save_kwargs["resolution"] = (dpi_x, dpi_y)
            save_kwargs["resolutionunit"] = "INCH"
        if icc_profile:
            save_kwargs["iccprofile"] = icc_profile

        tifffile.imwrite(tmp_tif, rgb, **save_kwargs)
        return tmp_tif

    raise ValueError(f"当前只支持 uint8/uint16 输出，当前 dtype={bgr.dtype}")

def _normalize_cv_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise FileNotFoundError("图像读取失败")

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    if img.ndim == 3 and img.shape[2] == 4:
        raise ValueError(
            "检测到 4 通道输入。当前保真链路不再默认白底合成。"
            "请先明确 alpha 的处理策略。"
        )

    raise ValueError(f"不支持的图像 shape: {img.shape}")

def read_cover_asset(image_path: str, max_side: int = 0) -> tuple[np.ndarray, SourceMeta]:
    path = Path(image_path)
    ext = path.suffix.lower()
    dpi_x, dpi_y, icc_profile = read_image_meta(path)

    if ext in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim == 3 and arr.shape[2] == 4:
            raise ValueError(
                "检测到 4 通道 TIFF。当前保真模式下，不再默认白底合成 alpha。"
                "请先明确是否需要保留透明通道，或另行改造导出链路。"
            )

        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"当前只支持 3 通道 TIFF，实际 shape={arr.shape}")

        if arr.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"当前只支持 uint8/uint16 TIFF，实际 dtype={arr.dtype}")

        # tifffile 读出来通常按 RGB，转成项目内部统一的 BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        depth = 16 if bgr.dtype == np.uint16 else 8
        meta = SourceMeta(
            input_path=str(path),
            input_ext=ext,
            input_is_psb=False,
            width=int(bgr.shape[1]),
            height=int(bgr.shape[0]),
            depth=depth,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            icc_profile=icc_profile,
        )
        return bgr, meta
    
    if ext not in {".psd", ".psb"}:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {path}")
        
        img = _normalize_cv_image(img)
        if img.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"当前只支持 uint8/uint16 输入，当前 dtype={img.dtype}")

        depth = 16 if img.dtype == np.uint16 else 8

        meta = SourceMeta(
            input_path=str(path),
            input_ext=ext,
            input_is_psb=False,
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            depth=depth,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            icc_profile=icc_profile,
        )
        return img, meta

    psd = PSDImage.open(path)
    psd_depth = int(getattr(psd, "depth", 8) or 8)

    if psd_depth == 16:
        raise RuntimeError(
            "当前代码仍然通过 psd_tools.composite().convert('RGB') 读取 PSB，"
            "这条路径会把 16-bit PSB 读成 8-bit。"
            "如果你要保真处理 16-bit PSB，请不要走当前分支；"
            "请先在 Photoshop 中导出 16-bit 中间格式，再送入本程序。"
        )

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
        depth=psd_depth,
    )
    return bgr, meta


def save_bgr_preview_png(path: str | Path, bgr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = to_preview_u8(bgr)
    ok = cv2.imwrite(str(path), preview)
    if not ok:
        raise RuntimeError(f"写入 PNG 失败: {path}")
    return path

def save_bgr_native_tiff(path: str | Path, bgr: np.ndarray, meta: SourceMeta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    save_kwargs = {
        "photometric": "rgb",
        "compression": None,
    }
    if meta.dpi_x and meta.dpi_y:
        save_kwargs["resolution"] = (meta.dpi_x, meta.dpi_y)
        save_kwargs["resolutionunit"] = "INCH"
    if meta.icc_profile:
        save_kwargs["iccprofile"] = meta.icc_profile

    tifffile.imwrite(path, rgb, **save_kwargs)
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


def save_bgr_as_flattened_psb_via_photoshop(path: str | Path, bgr: np.ndarray, dpi_x: float | None = None, dpi_y: float | None = None, icc_profile: bytes | None = None,) -> Path:
    from win32com.client import Dispatch

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_raster = write_temp_raster_for_photoshop(tmpdir, bgr, dpi_x=dpi_x, dpi_y=dpi_y, icc_profile=icc_profile)

        app = _dispatch_photoshop()
        app.DisplayDialogs = 3

        doc = app.Open(str(tmp_raster.resolve()))
        try:
            dpi_js = ""
            if dpi_x is not None and dpi_y is not None:
                # 只在输入明确带有 dpi 时，才把分辨率写回去
                dpi_js = f'doc.resizeImage(undefined, undefined, {dpi_x}, ResampleMethod.NONE);'

            jsx = rf"""
            #target photoshop
            (function () {{
                var doc = app.activeDocument;

                {dpi_js}

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