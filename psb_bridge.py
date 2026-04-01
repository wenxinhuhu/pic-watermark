from __future__ import annotations

import os
import uuid
import gc
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import tifffile
from PIL import Image
from psd_tools import PSDImage

def inspect_psdpsb_basic(image_path: str | Path) -> dict[str, Any]:
    path = Path(image_path)
    psd = PSDImage.open(path)

    width = int(getattr(psd, "width", 0) or 0)
    height = int(getattr(psd, "height", 0) or 0)
    depth = getattr(psd, "depth", None)
    color_mode = getattr(psd, "color_mode", None)

    return {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "width": width,
        "height": height,
        "depth": int(depth) if depth is not None else None,
        "color_mode": str(color_mode) if color_mode is not None else "",
    }


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


def _ps_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "\\\\")


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

def resolve_resolution_pair(
    dpi_x: float | None,
    dpi_y: float | None,
    default: float = 72.0,
) -> tuple[float, float]:
    rx = float(dpi_x) if dpi_x and dpi_x > 0 else float(default)
    ry = float(dpi_y) if dpi_y and dpi_y > 0 else float(default)
    return rx, ry

def memmap_tiff_rgb(path: str | Path, mode: str = "r"):
    arr = tifffile.memmap(str(path), mode=mode)
    if arr.ndim == 2:
        raise ValueError("当前不支持单通道 TIFF，请先转成 3 通道 RGB")
    if arr.ndim != 3:
        raise ValueError(f"当前只支持 3 维 TIFF，实际 shape={arr.shape}")
    if arr.shape[-1] == 3:
        pass
    elif arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"无法识别 TIFF 通道布局: shape={arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"当前只支持 uint8/uint16 TIFF，实际 dtype={arr.dtype}")
    return arr

def release_memmap(mm) -> None:
    if mm is None:
        return

    try:
        if hasattr(mm, "flush"):
            mm.flush()
    except Exception:
        pass

    try:
        mmap_obj = getattr(mm, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
    except Exception:
        pass

    try:
        base = getattr(mm, "base", None)
        mmap_obj = getattr(base, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
    except Exception:
        pass

    del mm
    gc.collect()

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
            "请改走大图桥接链路。"
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
        dpi_x=dpi_x,
        dpi_y=dpi_y,
        icc_profile=icc_profile,
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


def save_tiff_preview_png(
    tif_path: str | Path,
    out_png_path: str | Path,
    max_preview_side: int = 3000,
    *,
    max_side: int | None = None,
) -> Path:
    if max_side is not None:
        max_preview_side = int(max_side)

    tif_path = Path(tif_path)
    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)

    arr = None
    try:
        arr = memmap_tiff_rgb(tif_path)
        h, w = int(arr.shape[0]), int(arr.shape[1])
        step = max(1, int(math.ceil(max(h, w) / max(1, int(max_preview_side)))))
        sample = np.asarray(arr[::step, ::step, :])

        if sample.dtype == np.uint16:
            sample = np.rint(sample.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
        elif sample.dtype != np.uint8:
            raise ValueError(f"当前只支持 uint8/uint16 TIFF 预览，实际 dtype={sample.dtype}")

        Image.fromarray(sample, mode="RGB").save(out_png_path, format="PNG", compress_level=0)
        return out_png_path
    finally:
        release_memmap(arr)


def save_bgr_native_tiff(path: str | Path, bgr: np.ndarray, meta: SourceMeta) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    save_kwargs = {
        "photometric": "rgb",
        "compression": None,
        "bigtiff": bool(rgb.nbytes >= (4 * 1024**3)),
    }
    if meta.dpi_x and meta.dpi_y:
        save_kwargs["resolution"] = (meta.dpi_x, meta.dpi_y)
        save_kwargs["resolutionunit"] = "INCH"
    if meta.icc_profile:
        save_kwargs["iccprofile"] = meta.icc_profile
    tifffile.imwrite(path, rgb, **save_kwargs)
    return path


def clone_raster_file(
    src_path: str | Path,
    dst_path: str | Path,
    retries: int = 6,
    delay: float = 0.3,
) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc = None

    for i in range(retries):
        tmp_path = dst_path.parent / f"{dst_path.name}.tmp_{uuid.uuid4().hex}"
        try:
            if dst_path.exists():
                try:
                    dst_path.unlink()
                except PermissionError:
                    gc.collect()
                    time.sleep(delay * (i + 1))
                    if dst_path.exists():
                        dst_path.unlink()

            with open(src_path, "rb") as rf, open(tmp_path, "wb") as wf:
                shutil.copyfileobj(rf, wf, length=16 * 1024 * 1024)
                wf.flush()
                os.fsync(wf.fileno())

            os.replace(tmp_path, dst_path)
            return dst_path

        except PermissionError as exc:
            last_exc = exc
            gc.collect()
            time.sleep(delay * (i + 1))
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

    raise RuntimeError(f"目标文件无法写入或被占用: {dst_path}") from last_exc


def _dispatch_photoshop():
    try:
        import pythoncom  # type: ignore
        pythoncom.CoInitialize()
    except Exception:
        pass
    try:
        from win32com.client import Dispatch  # type: ignore
        app = Dispatch("Photoshop.Application")
        app.DisplayDialogs = 3
        return app
    except Exception as exc:
        raise RuntimeError(
            "无法连接 Photoshop COM（Photoshop.Application）。"
            "通常是因为本机未安装 Photoshop，或 Photoshop 的 COM 注册不可用。"
        ) from exc


def _write_rgb_tiff(tile_path: str | Path, rgb: np.ndarray, dpi_x: float | None, dpi_y: float | None) -> Path:
    tile_path = Path(tile_path)
    tile_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "photometric": "rgb",
        "compression": None,
        "bigtiff": False,
    }
    if dpi_x and dpi_y:
        kwargs["resolution"] = (dpi_x, dpi_y)
        kwargs["resolutionunit"] = "INCH"
    tifffile.imwrite(str(tile_path), rgb, **kwargs)
    return tile_path


def _read_rgb_tiff(path: str | Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"当前仅支持 3 通道 TIFF tile，实际 shape={arr.shape}")
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"当前仅支持 uint8/uint16 TIFF tile，实际 dtype={arr.dtype}")
    return arr


def _photoshop_open(path: str | Path):
    app = _dispatch_photoshop()
    return app, app.Open(str(Path(path).resolve()))

def read_psdpsb_resolution_via_photoshop(image_path: str | Path) -> tuple[float | None, float | None]:
    app, doc = _photoshop_open(image_path)
    try:
        res = getattr(doc, "Resolution", None)
        if res is None:
            res = getattr(doc, "resolution", None)
        if res is None:
            return None, None
        res = float(res)
        return res, res
    except Exception:
        return None, None
    finally:
        try:
            doc.Close(2)
        except Exception:
            pass

def _export_active_psd_tile_to_tiff(
    app,
    out_tif_path: str | Path,
    x: int,
    y: int,
    w: int,
    h: int,
) -> Path:
    out_tif_path = Path(out_tif_path)
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)

    jsx = rf'''
#target photoshop
(function () {{
    var src = app.activeDocument;
    if (src.mode != DocumentMode.RGB) {{
        throw new Error("当前仅支持 RGB 模式 PSB/PSD 的 tile 导出");
    }}

    var left = UnitValue({int(x)}, "px");
    var top = UnitValue({int(y)}, "px");
    var right = UnitValue({int(x + w)}, "px");
    var bottom = UnitValue({int(y + h)}, "px");

    var tileDoc = null;
    try {{
        tileDoc = src.duplicate("wm_tile", false);
    }} catch (e1) {{
        tileDoc = src.duplicate();
    }}

    app.activeDocument = tileDoc;
    tileDoc.crop([left, top, right, bottom]);

    try {{
        tileDoc.flatten();
    }} catch (e2) {{}}

    var outFile = new File("{_ps_path(out_tif_path)}");
    var opts = new TiffSaveOptions();
    opts.layers = false;
    opts.embedColorProfile = true;
    opts.imageCompression = TIFFEncoding.NONE;

    tileDoc.saveAs(outFile, opts, true, Extension.LOWERCASE);
    tileDoc.close(SaveOptions.DONOTSAVECHANGES);

    app.activeDocument = src;
}})();
'''
    app.DoJavaScript(jsx)

    if not out_tif_path.exists():
        raise RuntimeError(f"Photoshop 未成功导出 tile: {out_tif_path}")
    return out_tif_path

class _PsbAssembler:
    def __init__(self, width: int, height: int, bit_depth: int, resolution: float, doc_name: str) -> None:
        self.width = int(width)
        self.height = int(height)
        self.bit_depth = int(bit_depth)
        self.resolution = float(resolution)
        self.doc_name = doc_name.replace('"', '_')
        self.app = _dispatch_photoshop()

    def _run_js(self, jsx: str) -> None:
        self.app.DoJavaScript(jsx)

    def create(self) -> None:
        bits_expr = "BitsPerChannelType.SIXTEEN" if self.bit_depth > 8 else "BitsPerChannelType.EIGHT"
        jsx = rf'''
#target photoshop
(function () {{
    app.documents.add(
        UnitValue({self.width}, "px"),
        UnitValue({self.height}, "px"),
        {self.resolution},
        "{self.doc_name}",
        NewDocumentMode.RGB,
        DocumentFill.WHITE,
        1.0,
        {bits_expr}
    );
}})();
'''
        self._run_js(jsx)

    def paste_tile(self, tile_path: str | Path, x0: int, y0: int) -> None:
        jsx = rf'''
#target photoshop
(function () {{
    var dst = app.documents.getByName("{self.doc_name}");
    app.activeDocument = dst;
    var tileDoc = app.open(new File("{_ps_path(tile_path)}"));
    app.activeDocument = tileDoc;
    tileDoc.selection.selectAll();
    tileDoc.selection.copy();
    app.activeDocument = dst;
    dst.paste();
    var lyr = dst.activeLayer;
    var b = lyr.bounds;
    var left = b[0].as("px");
    var top = b[1].as("px");
    lyr.translate(UnitValue({int(x0)} - left, "px"), UnitValue({int(y0)} - top, "px"));
    dst.flatten();
    app.activeDocument = tileDoc;
    tileDoc.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = dst;
}})();
'''
        self._run_js(jsx)

    def save_psb(self, out_psb_path: str | Path) -> Path:
        out_psb_path = Path(out_psb_path)
        out_psb_path.parent.mkdir(parents=True, exist_ok=True)
        jsx = rf'''
#target photoshop
(function () {{
    var dst = app.documents.getByName("{self.doc_name}");
    app.activeDocument = dst;
    var theFile = new File("{_ps_path(out_psb_path)}");
    var s2t = function (s) {{ return app.stringIDToTypeID(s); }};
    var descriptor = new ActionDescriptor();
    var descriptor2 = new ActionDescriptor();
    descriptor.putObject(s2t("as"), s2t("largeDocumentFormat"), descriptor2);
    descriptor.putPath(s2t("in"), theFile);
    descriptor.putBoolean(s2t("copy"), false);
    descriptor.putBoolean(s2t("lowerCase"), true);
    executeAction(s2t("save"), descriptor, DialogModes.NO);
}})();
'''
        self._run_js(jsx)
        if not out_psb_path.exists():
            raise RuntimeError(f"Photoshop 未成功生成 PSB: {out_psb_path}")
        return out_psb_path

    def close(self) -> None:
        jsx = rf'''
#target photoshop
(function () {{
    try {{
        var dst = app.documents.getByName("{self.doc_name}");
        app.activeDocument = dst;
        dst.close(SaveOptions.DONOTSAVECHANGES);
    }} catch (e) {{}}
}})();
'''
        try:
            self._run_js(jsx)
        except Exception:
            pass


def convert_existing_tiff_to_psb(
    input_tif_path: str | Path,
    out_psb_path: str | Path,
    tile_size: int = 1024,
) -> Path:
    input_tif_path = Path(input_tif_path)
    out_psb_path = Path(out_psb_path)
    rgb = memmap_tiff_rgb(input_tif_path)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    bit_depth = 16 if rgb.dtype == np.uint16 else 8
    dpi_x, dpi_y, _icc = read_image_meta(input_tif_path)
    resolution = float(dpi_x or dpi_y or 72.0)
    tile_cols = math.ceil(width / tile_size)
    tile_rows = math.ceil(height / tile_size)
    tile_total = tile_cols * tile_rows
    doc_name = out_psb_path.stem or "wm_output_psb"
    assembler = _PsbAssembler(width, height, bit_depth, resolution, doc_name)
    assembler.create()
    try:
        with tempfile.TemporaryDirectory(prefix="psb_tiles_") as tmpdir:
            tmpdir = Path(tmpdir)
            tile_index = 0
            for y0 in range(0, height, tile_size):
                hh = min(tile_size, height - y0)
                for x0 in range(0, width, tile_size):
                    ww = min(tile_size, width - x0)
                    tile_rgb = np.asarray(rgb[y0:y0 + hh, x0:x0 + ww, :])
                    tile_path = tmpdir / f"tile_{tile_index:06d}.tif"
                    _write_rgb_tiff(tile_path, tile_rgb, dpi_x, dpi_y)
                    assembler.paste_tile(tile_path, x0, y0)
                    tile_index += 1
                    if tile_index == 1 or tile_index % 10 == 0 or tile_index == tile_total:
                        print(f"[INFO] 已写入 tile {tile_index}/{tile_total} ({x0},{y0},{ww},{hh})")
        return assembler.save_psb(out_psb_path)
    finally:
        assembler.close()



def save_flattened_tiff_as_psb_via_photoshop(
    input_tif_path: str | Path,
    out_psb_path: str | Path,
    tile_size: int = 1024,
    try_direct_first: bool = True,
    dpi_x: float | None = None,
    dpi_y: float | None = None,
) -> Path:
    input_tif_path = Path(input_tif_path)
    out_psb_path = Path(out_psb_path)

    # dpi_x, dpi_y, _icc = read_image_meta(input_tif_path)
    # resolution = float(dpi_x or dpi_y or 72.0)
    read_dpi_x, read_dpi_y, _icc = read_image_meta(input_tif_path)

    # 显式传入优先；没传再回退到 TIFF 文件自身元数据
    dpi_x = dpi_x if dpi_x is not None else read_dpi_x
    dpi_y = dpi_y if dpi_y is not None else read_dpi_y

    rx, ry = resolve_resolution_pair(dpi_x, dpi_y, default=72.0)
    resolution = rx

    if try_direct_first:
        try:
            app, doc = _photoshop_open(input_tif_path)
            try:
                jsx = rf'''
#target photoshop
(function () {{
    var doc = app.activeDocument;
    doc.resizeImage(undefined, undefined, {resolution}, ResampleMethod.NONE);

    var theFile = new File("{_ps_path(out_psb_path)}");
    var s2t = function (s) {{ return app.stringIDToTypeID(s); }};
    var descriptor = new ActionDescriptor();
    var descriptor2 = new ActionDescriptor();
    descriptor.putObject(s2t("as"), s2t("largeDocumentFormat"), descriptor2);
    descriptor.putPath(s2t("in"), theFile);
    descriptor.putBoolean(s2t("copy"), true);
    descriptor.putBoolean(s2t("lowerCase"), true);
    executeAction(s2t("save"), descriptor, DialogModes.NO);
}})();
'''
                app.DoJavaScript(jsx)
            finally:
                doc.Close(2)
            if out_psb_path.exists():
                return out_psb_path
        except Exception:
            pass
    return convert_existing_tiff_to_psb(input_tif_path, out_psb_path, tile_size=tile_size)


def save_bgr_as_flattened_psb_via_photoshop(
    path: str | Path,
    bgr: np.ndarray,
    dpi_x: float | None = None,
    dpi_y: float | None = None,
    icc_profile: bytes | None = None,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        if bgr.dtype == np.uint8:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tmp_path = tmpdir / "wm_tmp.png"
            kwargs = {"format": "PNG", "compress_level": 0}
            if dpi_x and dpi_y:
                kwargs["dpi"] = (dpi_x, dpi_y)
            if icc_profile:
                kwargs["icc_profile"] = icc_profile
            Image.fromarray(rgb).save(tmp_path, **kwargs)
        elif bgr.dtype == np.uint16:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tmp_path = tmpdir / "wm_tmp.tif"
            kwargs = {"photometric": "rgb", "compression": None, "bigtiff": bool(rgb.nbytes >= (4 * 1024**3))}
            if dpi_x and dpi_y:
                kwargs["resolution"] = (dpi_x, dpi_y)
                kwargs["resolutionunit"] = "INCH"
            if icc_profile:
                kwargs["iccprofile"] = icc_profile
            tifffile.imwrite(tmp_path, rgb, **kwargs)
        else:
            raise ValueError(f"当前只支持 uint8/uint16 输出，当前 dtype={bgr.dtype}")
        return save_flattened_tiff_as_psb_via_photoshop(tmp_path, out_path, tile_size=1024,dpi_x=dpi_x, dpi_y=dpi_y)

def export_flattened_psdpsb_to_tiff_via_photoshop(
    image_path: str | Path,
    out_tif_path: str | Path,
    tile_size: int = 1024,
) -> tuple[Path, SourceMeta]:
    image_path = Path(image_path)
    out_tif_path = Path(out_tif_path)
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)

    psd = PSDImage.open(image_path)
    width = int(getattr(psd, "width", 0))
    height = int(getattr(psd, "height", 0))
    depth = int(getattr(psd, "depth", 8) or 8)
    dtype = np.uint16 if depth >= 16 else np.uint8

    app, src_doc = _photoshop_open(image_path)
    try:
        dpi_x, dpi_y, icc_profile = read_image_meta(image_path)

        try:
            res = getattr(src_doc, "Resolution", None)
            if res is None:
                res = getattr(src_doc, "resolution", None)
            if res is not None:
                res = float(res)
                dpi_x = res
                dpi_y = res
        except Exception:
            pass

        with tempfile.TemporaryDirectory(prefix="psb_export_") as tmpdir:
            tmpdir = Path(tmpdir)

            first_x0, first_y0 = 0, 0
            first_ww = min(tile_size, width)
            first_hh = min(tile_size, height)
            first_tile_path = tmpdir / "tile_first.tif"

            _export_active_psd_tile_to_tiff(
                app,
                first_tile_path,
                first_x0,
                first_y0,
                first_ww,
                first_hh,
            )

            if dpi_x is None or dpi_y is None or icc_profile is None:
                tile_dpi_x, tile_dpi_y, tile_icc = read_image_meta(first_tile_path)
                dpi_x = dpi_x or tile_dpi_x
                dpi_y = dpi_y or tile_dpi_y
                icc_profile = icc_profile or tile_icc

            with tifffile.TiffWriter(str(out_tif_path), bigtiff=True) as tw:
                write_kwargs = {
                    "data": None,
                    "shape": (height, width, 3),
                    "dtype": dtype,
                    "photometric": "rgb",
                    "compression": None,
                    "metadata": None,
                }
                if dpi_x and dpi_y:
                    write_kwargs["resolution"] = (dpi_x, dpi_y)
                    write_kwargs["resolutionunit"] = "INCH"
                if icc_profile:
                    write_kwargs["iccprofile"] = icc_profile
                tw.write(**write_kwargs)

            # rgb = tifffile.memmap(str(out_tif_path), mode="r+")

            # first_tile_rgb = _read_rgb_tiff(first_tile_path)
            # rgb[first_y0:first_y0 + first_hh, first_x0:first_x0 + first_ww, :] = first_tile_rgb

            # tile_index = 1
            # tile_total = math.ceil(width / tile_size) * math.ceil(height / tile_size)

            # if tile_index == 1 or tile_index == tile_total:
            #     print(f"[INFO] 已导出 tile {tile_index}/{tile_total} ({first_x0},{first_y0},{first_ww},{first_hh})")

            # for y0 in range(0, height, tile_size):
            #     hh = min(tile_size, height - y0)
            #     for x0 in range(0, width, tile_size):
            #         ww = min(tile_size, width - x0)

            #         if x0 == 0 and y0 == 0:
            #             continue

            #         tile_path = tmpdir / f"tile_{tile_index:06d}.tif"
            #         _export_active_psd_tile_to_tiff(app, tile_path, x0, y0, ww, hh)
            #         tile_rgb = _read_rgb_tiff(tile_path)
            #         rgb[y0:y0 + hh, x0:x0 + ww, :] = tile_rgb

            #         tile_index += 1
            #         if tile_index % 10 == 0 or tile_index == tile_total:
            #             print(f"[INFO] 已导出 tile {tile_index}/{tile_total} ({x0},{y0},{ww},{hh})")

            # if hasattr(rgb, "flush"):
            #     rgb.flush()
            rgb = None
            try:
                rgb = tifffile.memmap(str(out_tif_path), mode="r+")

                first_tile_rgb = _read_rgb_tiff(first_tile_path)
                rgb[first_y0:first_y0 + first_hh, first_x0:first_x0 + first_ww, :] = first_tile_rgb

                tile_index = 1
                tile_total = math.ceil(width / tile_size) * math.ceil(height / tile_size)

                if tile_index == 1 or tile_index == tile_total:
                    print(f"[INFO] 已导出 tile {tile_index}/{tile_total} ({first_x0},{first_y0},{first_ww},{first_hh})")

                for y0 in range(0, height, tile_size):
                    hh = min(tile_size, height - y0)
                    for x0 in range(0, width, tile_size):
                        ww = min(tile_size, width - x0)

                        if x0 == 0 and y0 == 0:
                            continue

                        tile_path = tmpdir / f"tile_{tile_index:06d}.tif"
                        _export_active_psd_tile_to_tiff(app, tile_path, x0, y0, ww, hh)
                        tile_rgb = _read_rgb_tiff(tile_path)
                        rgb[y0:y0 + hh, x0:x0 + ww, :] = tile_rgb

                        tile_index += 1
                        if tile_index % 10 == 0 or tile_index == tile_total:
                            print(f"[INFO] 已导出 tile {tile_index}/{tile_total} ({x0},{y0},{ww},{hh})")
            finally:
                release_memmap(rgb)

    finally:
        try:
            src_doc.Close(2)
        except Exception:
            pass

    meta = SourceMeta(
        input_path=str(image_path),
        input_ext=image_path.suffix.lower(),
        input_is_psb=True,
        width=width,
        height=height,
        depth=depth,
        dpi_x=dpi_x,
        dpi_y=dpi_y,
        icc_profile=icc_profile,
    )
    return out_tif_path, meta

def prepare_large_raster_input(
    image_path: str | Path,
    out_dir: str | Path,
    tile_size: int = 1024,
) -> tuple[Path, SourceMeta]:
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = image_path.suffix.lower()
    out_tif_path = out_dir / f"{image_path.stem}_flattened_input.tif"

    if ext in {".psd", ".psb"}:
        return export_flattened_psdpsb_to_tiff_via_photoshop(
            image_path,
            out_tif_path,
            tile_size=tile_size,
        )

    if ext in {".tif", ".tiff"}:
        clone_raster_file(image_path, out_tif_path)

        arr = None
        try:
            arr = memmap_tiff_rgb(out_tif_path)
            dpi_x, dpi_y, icc_profile = read_image_meta(out_tif_path)
            meta = SourceMeta(
                input_path=str(image_path),
                input_ext=ext,
                input_is_psb=False,
                width=int(arr.shape[1]),
                height=int(arr.shape[0]),
                depth=16 if arr.dtype == np.uint16 else 8,
                dpi_x=dpi_x,
                dpi_y=dpi_y,
                icc_profile=icc_profile,
            )
            return out_tif_path, meta
        finally:
            release_memmap(arr)

    bgr, meta = read_cover_asset(str(image_path), max_side=0)
    save_bgr_native_tiff(out_tif_path, bgr, meta)
    return out_tif_path, meta
