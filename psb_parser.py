from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Any

from PIL import Image
from psd_tools import PSDImage


@dataclass
class LayerMeta:
    index: int
    path: str
    name: str
    kind: str
    visible: bool
    is_group: bool
    opacity: int | None
    blend_mode: str | None
    bbox: tuple[int, int, int, int] | None


def safe_name(text: str) -> str:
    text = (text or "").strip() or "unnamed"
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120]


def is_group_layer(layer: Any) -> bool:
    try:
        return bool(layer.is_group())
    except Exception:
        return False

#递归遍历图层树
def walk_layers(layers: Any, parent_path: str = "") -> Iterator[tuple[str, Any]]:
    for layer in layers:
        layer_name = str(getattr(layer, "name", "") or "unnamed")
        current_path = f"{parent_path}/{layer_name}" if parent_path else layer_name
        yield current_path, layer

        if is_group_layer(layer):
            yield from walk_layers(layer, current_path)


def get_bbox_tuple(bbox_obj: Any) -> tuple[int, int, int, int] | None:
    if bbox_obj is None:
        return None

    try:
        return (int(bbox_obj.x1), int(bbox_obj.y1), int(bbox_obj.x2), int(bbox_obj.y2))
    except Exception:
        pass

    try:
        values = tuple(int(v) for v in bbox_obj)
        if len(values) == 4:
            return values
    except Exception:
        pass

    return None


def export_psb(
    psb_path: str | Path,
    out_dir: str | Path = "psb_dump",
    export_layers: bool = False,
    only_visible_layers: bool = True,
    make_preview: bool = True,
    preview_max_side: int = 2000,
) -> dict:
    psb_path = Path(psb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not psb_path.exists():
        raise FileNotFoundError(f"文件不存在: {psb_path}")

    if psb_path.suffix.lower() not in {".psb", ".psd"}:
        raise ValueError(f"只支持 .psb / .psd，当前文件: {psb_path.name}")

    psd = PSDImage.open(psb_path)

    doc_info = {
        "file": str(psb_path),
        "width": int(getattr(psd, "width", 0) or 0),
        "height": int(getattr(psd, "height", 0) or 0),
        "color_mode": str(getattr(psd, "color_mode", None)),
        "depth": getattr(psd, "depth", None),
        "layer_count_total": 0,
        "layer_count_groups": 0,
        "layer_count_leaf": 0,
    }

    composite_pil = psd.composite()
    composite_png = out_dir / "composite.png"
    composite_pil.save(composite_png)

    preview_path = None
    if make_preview:
        preview = composite_pil.copy()
        preview.thumbnail((preview_max_side, preview_max_side))
        preview_path = out_dir / "composite_preview.jpg"
        preview.convert("RGB").save(preview_path, quality=90)

    layer_dump_dir = out_dir / "layers"
    if export_layers:
        layer_dump_dir.mkdir(parents=True, exist_ok=True)

    layer_list: list[LayerMeta] = []
    export_errors: list[dict] = []

    for idx, (layer_path, layer) in enumerate(walk_layers(psd)):
        group_flag = is_group_layer(layer)
        visible = bool(getattr(layer, "visible", True))
        bbox = get_bbox_tuple(getattr(layer, "bbox", None))

        meta = LayerMeta(
            index=idx,
            path=layer_path,
            name=str(getattr(layer, "name", "") or ""),
            kind=str(getattr(layer, "kind", "") or ""),
            visible=visible,
            is_group=group_flag,
            opacity=getattr(layer, "opacity", None),
            blend_mode=str(getattr(layer, "blend_mode", None)),
            bbox=bbox,
        )
        layer_list.append(meta)

        doc_info["layer_count_total"] += 1
        if group_flag:
            doc_info["layer_count_groups"] += 1
        else:
            doc_info["layer_count_leaf"] += 1

        if not export_layers:
            continue
        if group_flag:
            continue
        if only_visible_layers and not visible:
            continue

        try:
            pil_img = None

            try:
                pil_img = layer.topil()
            except Exception:
                pil_img = None
            if pil_img is None:
                try:
                    pil_img = layer.composite()
                except Exception:
                    pil_img = None

            if pil_img is None:
                export_errors.append(
                    {
                        "index": idx,
                        "path": layer_path,
                        "reason": "topil/composite returned None",
                    }
                )
                continue

            filename = f"{idx:03d}_{safe_name(meta.name)}.png"
            pil_img.save(layer_dump_dir / filename)

        except Exception as exc:
            export_errors.append(
                {
                    "index": idx,
                    "path": layer_path,
                    "reason": repr(exc),
                }
            )

    meta_json = {
        "document": doc_info,
        "layers": [asdict(x) for x in layer_list],
        "export_errors": export_errors,
    }

    layers_json_path = out_dir / "layers.json"
    layers_json_path.write_text(
        json.dumps(meta_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "document": doc_info,
        "composite_png": str(composite_png),
        "preview_jpg": str(preview_path) if preview_path else None,
        "layers_json": str(layers_json_path),
        "layers_dir": str(layer_dump_dir) if export_layers else None,
        "export_errors_count": len(export_errors),
    }


if __name__ == "__main__":
    result = export_psb(
        psb_path=r"D:\pic-watermark\pic-watermark\image\连年有余实验用图-不可外传.psb",
        out_dir=r"D:\pic-watermark\pic-watermark\psb_dump",
        export_layers=False,       # 首次建议 False，先验证解析是否成功
        only_visible_layers=True,
        make_preview=True,
        preview_max_side=2000,
    )

    print("解析完成")
    print(json.dumps(result["document"], ensure_ascii=False, indent=2))
    print("合成图:", result["composite_png"])
    print("预览图:", result["preview_jpg"])
    print("图层信息:", result["layers_json"])
    print("图层导出错误数:", result["export_errors_count"])