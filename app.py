from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from datetime import datetime

from flask import (
    Flask,
    render_template,
    send_from_directory,
    request,
    redirect,
    url_for,
    flash,
    abort,
)

import demo as demo_mod
from psb_bridge import inspect_psdpsb_basic
from tardos import UserRecord
import uuid
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "dev-secret-for-flask-session"

LARGE_RASTER_SIZE_THRESHOLD_BYTES = 512 * 1024 * 1024
ULTRA_LARGE_PSB_FORCE_LARGE_RASTER_BYTES = 2 * 1024 * 1024 * 1024


def _normalize_image_suffix(image_path: str | None) -> str:
    if not image_path:
        return ""
    return Path(image_path).suffix.lower()


def _get_file_size(image_path: str | None) -> int | None:
    if not image_path:
        return None
    try:
        return Path(image_path).stat().st_size
    except Exception:
        return None


def inspect_processing_mode(image_path: str | None) -> dict[str, Any]:
    suffix = _normalize_image_suffix(image_path)
    size_bytes = _get_file_size(image_path)
    info: dict[str, Any] = {
        "image_path": image_path,
        "suffix": suffix,
        "size_bytes": size_bytes,
        "is_psb_family": suffix in {".psb", ".psd"},
        "depth": None,
        "width": None,
        "height": None,
        "use_large_raster": False,
        "reason": "default",
        "requested_mode": "auto",
    }

    if not image_path:
        return info

    if info["is_psb_family"]:
        try:
            meta = inspect_psdpsb_basic(image_path)
            info["depth"] = meta.get("depth")
            info["width"] = meta.get("width")
            info["height"] = meta.get("height")
        except Exception as exc:
            info["inspect_error"] = str(exc)
            return info

        if size_bytes is not None and size_bytes >= ULTRA_LARGE_PSB_FORCE_LARGE_RASTER_BYTES:
            info["use_large_raster"] = True
            info["reason"] = f"ultra large PSB/PSD >= {ULTRA_LARGE_PSB_FORCE_LARGE_RASTER_BYTES // (1024 * 1024)} MB"
            return info

        if int(info["depth"] or 0) >= 16:
            info["use_large_raster"] = True
            info["reason"] = "16-bit PSB/PSD"
            return info

        if size_bytes is not None and size_bytes >= LARGE_RASTER_SIZE_THRESHOLD_BYTES:
            info["use_large_raster"] = True
            info["reason"] = f"large PSB/PSD >= {LARGE_RASTER_SIZE_THRESHOLD_BYTES // (1024 * 1024)} MB"
            return info

        return info

    if suffix in {".tif", ".tiff"} and size_bytes is not None and size_bytes >= LARGE_RASTER_SIZE_THRESHOLD_BYTES:
        info["use_large_raster"] = True
        info["reason"] = f"large TIFF >= {LARGE_RASTER_SIZE_THRESHOLD_BYTES // (1024 * 1024)} MB"

    return info

def apply_processing_mode_override(route_info: dict[str, Any], requested_mode: str | None, image_path: str | None) -> dict[str, Any]:
    mode = (requested_mode or "auto").strip() or "auto"
    info = dict(route_info)
    info["requested_mode"] = mode

    if mode == "auto":
        return info

    suffix = _normalize_image_suffix(image_path)
    info["use_large_raster"] = False

    if mode == "default":
        info["reason"] = "forced default"
        return info

    if mode == "large_raster":
        if suffix not in {".psb", ".psd", ".tif", ".tiff"}:
            raise ValueError("large_raster 仅支持 .psb / .psd / .tif / .tiff 输入")
        info["use_large_raster"] = True
        info["reason"] = "forced large_raster"
        return info

    raise ValueError(f"未知处理模式: {mode}")


def select_single_user(users: list[dict[str, str]], requested_num_users: int) -> dict[str, str]:
    if not users:
        raise ValueError("当前没有可用用户，请先添加用户")
    if requested_num_users != 1:
        app.logger.warning("single-file mode only uses the first configured user; requested num_users=%s", requested_num_users)
    return users[0]


def write_unified_report(out_path: Path, report: dict[str, Any] | None = None, report_path: str | Path | None = None) -> dict[str, Any]:
    out_path.mkdir(parents=True, exist_ok=True)
    if report is None:
        if report_path is None:
            raise ValueError("report 和 report_path 不能同时为空")
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    target = out_path / "report.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def ensure_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        users = [{"user_id": u.user_id, "user_key": u.user_key} for u in demo_mod.USER_DB]
        USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")


def load_users() -> list[dict[str, str]]:
    ensure_data()
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return [{"user_id": u.user_id, "user_key": u.user_key} for u in demo_mod.USER_DB]


def save_users(users: list[dict[str, str]]) -> None:
    ensure_data()
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    demo_mod.USER_DB = [UserRecord(user_id=u["user_id"], user_key=u["user_key"]) for u in users]


def resolve_out_dir(out_dir: str | None, default: Path | None = None) -> Path:
    if out_dir:
        target = (BASE_DIR / out_dir).resolve()
    else:
        target = (default or RESULTS_DIR).resolve()

    try:
        target.relative_to(BASE_DIR.resolve())
    except ValueError:
        abort(404)

    return target


def _result_filename(path_value: Any, available_files: list[str]) -> str | None:
    if not path_value:
        return None
    try:
        name = Path(str(path_value)).name
    except Exception:
        return None
    return name if name in available_files else None


def build_single_file_view(report: dict[str, Any] | None, available_files: list[str]) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    
    outputs = report.get("outputs", {}) or {}
    if outputs.get("users") or report.get("user_results"):
        return None
    
    mode = str(report.get("mode") or "")
    if mode not in {
        "large_single_file_tiled",
        "large_single_file_extract_tiled",
    }:
        return None

    return {
        "mode": mode,
        "route": report.get("app_route") or mode,
        "reason": report.get("route_reason") or report.get("reason") or "single-file mode",
        "user_id": report.get("selected_user_id") or report.get("user_id"),
        "requested_num_users": report.get("requested_num_users"),
        "embedded_text": report.get("display_text") or report.get("embedded_text"),
        "extracted_text": report.get("verify_extracted_text") or report.get("extracted_text"),
        "text_ok": report.get("verify_text_ok") if report.get("verify_text_ok") is not None else report.get("text_ok"),
        "fingerprint_ber": report.get("fingerprint_ber"),
        "tile_size": report.get("tile_size"),
        "tile_count": report.get("tile_count") or report.get("tile_count_used"),
        "input_depth": report.get("input_depth"),
        "width": report.get("width"),
        "height": report.get("height"),
        "preview_png": _result_filename(report.get("output_png"), available_files),
        "output_tif": _result_filename(report.get("output_tif"), available_files),
        "output_psb": _result_filename(report.get("output_psb"), available_files),
        "plan_json": _result_filename(report.get("plan_json"), available_files),
        "report_json": _result_filename(report.get("report_json"), available_files),
    }


@app.route("/")
def index() -> str:
    users = load_users()

    out_dir_param = request.args.get("out_dir")
    out_path = resolve_out_dir(out_dir_param, default=RESULTS_DIR)

    images = sorted([p.name for p in out_path.glob("*.png")]) if out_path.exists() else []

    report = None
    report_path = out_path / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = {"error": "无法解析 report.json"}

    extract_result = None
    if request.args.get("extract_text") is not None:
        extract_result = {
            "image_path": request.args.get("extract_image_path", ""),
            "input_is_psb": request.args.get("extract_input_is_psb", ""),
            "input_depth": request.args.get("extract_input_depth", ""),
            "display_bits_len": request.args.get("extract_display_bits_len", ""),
            "extracted_text": request.args.get("extract_text", ""),
            "text_ok": request.args.get("extract_ok", "false") == "true",
            "route": request.args.get("extract_route", "default"),
            "route_reason": request.args.get("extract_route_reason", "default"),
        }

    return render_template(
        "index.html",
        users=users,
        images=images,
        report=report,
        current_out_dir=str(out_path.relative_to(BASE_DIR)),
        extract_result=extract_result,
        large_raster_threshold_mb=LARGE_RASTER_SIZE_THRESHOLD_BYTES // (1024 * 1024),
    )


@app.route("/extract_text", methods=["POST"])
def extract_text():
    out_dir_form = request.form.get("out_dir", "results").strip() or "results"

    master_key = request.form.get("extract_master_key", "server-master-key-2026").strip()
    content_id = request.form.get("extract_content_id", "asset-demo-001").strip()
    delta = float(request.form.get("extract_delta", 8.0))
    repeats = int(request.form.get("extract_repeats", 2))
    code_length = int(request.form.get("extract_code_length", 512))
    delta_mode = request.form.get("extract_delta_mode", "normalized_8bit")
    psb_max_side = int(request.form.get("extract_psb_max_side", 0))
    tile_size = int(request.form.get("extract_tile_size", 2048) or 2048)
    route_mode = request.form.get("extract_route_mode", "auto")

    image_path_text = request.form.get("extract_image_path", "").strip()
    image_file = request.files.get("extract_image_file")

    try:
        image_path = None

        if image_file and image_file.filename:
            upload_dir = DATA_DIR / "extract_uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            safe_name = secure_filename(image_file.filename) or "upload.png"
            suffix = Path(safe_name).suffix or ".png"
            saved_path = upload_dir / f"extract_{uuid.uuid4().hex}{suffix}"
            image_file.save(str(saved_path))
            image_path = str(saved_path)

        elif image_path_text:
            image_path = image_path_text

        else:
            flash("请上传一张带水印图片，或填写图片路径", "danger")
            return redirect(url_for("index", out_dir=out_dir_form))

        result = extract_display_text_from_image(
            image_path=image_path,
            master_key=master_key,
            content_id=content_id,
            delta=delta,
            repeats=repeats,
            code_length=code_length,
            delta_mode=delta_mode,
            psb_max_side=psb_max_side,
            out_dir=out_dir_form,
            tile_size=tile_size,
            route_mode=route_mode,
        )

        flash("字符串提取完成", "success")

        return redirect(
            url_for(
                "index",
                out_dir=out_dir_form,
                extract_text=result["extracted_text"],
                extract_ok=str(result["text_ok"]).lower(),
                extract_image_path=result["image_path"],
                extract_input_is_psb=str(result["input_is_psb"]).lower(),
                extract_input_depth=result["input_depth"],
                extract_display_bits_len=result["display_bits_len"],
                extract_route=result.get("route", "default"),
                extract_route_reason=result.get("route_reason", "default"),
            )
        )

    except Exception as exc:
        flash(f"字符串提取失败: {exc}", "danger")
        return redirect(url_for("index", out_dir=out_dir_form))


@app.route("/users/add", methods=["POST"])
def add_user():
    user_id = request.form.get("user_id", "").strip()
    user_key = request.form.get("user_key", "").strip()
    if not user_id or not user_key:
        flash("user_id 和 user_key 不能为空", "danger")
        return redirect(url_for("index"))

    users = load_users()
    if any(u["user_id"] == user_id for u in users):
        flash("用户ID 已存在", "warning")
        return redirect(url_for("index"))

    users.append({"user_id": user_id, "user_key": user_key})
    save_users(users)
    flash("用户已添加", "success")
    return redirect(url_for("index"))


@app.route("/users/delete", methods=["POST"])
def delete_user():
    user_id = request.form.get("user_id", "").strip()
    users = load_users()
    users = [u for u in users if u["user_id"] != user_id]
    save_users(users)
    flash("用户已删除", "success")
    return redirect(url_for("index"))


@app.route("/run", methods=["POST"])
def run():
    image = request.form.get("image") or None
    num_users = int(request.form.get("num_users", 8))
    code_length = int(request.form.get("code_length", 320))
    colluders = request.form.get("colluders", "").strip()
    delta = float(request.form.get("delta", 8.0))
    repeats = int(request.form.get("repeats", 2))
    delta_mode = request.form.get("delta_mode", "normalized_8bit")
    master_key = request.form.get("master_key", "server-master-key-2026")
    content_id = request.form.get("content_id", "asset-demo-001")
    seed = int(request.form.get("seed", 2026))
    cover_size = int(request.form.get("cover_size", 512))
    processing_mode = request.form.get("processing_mode", "auto")

    emit_psb = request.form.get("emit_psb") == "on"
    psb_max_side = int(request.form.get("psb_max_side", 0))
    tile_size = int(request.form.get("tile_size", 2048)) if request.form.get("tile_size") else 2048
    
    users = load_users()
    if num_users < 1 or num_users > len(users):
        flash(f"num_users 必须在 1 和 {len(users)} 之间", "danger")
        return redirect(url_for("index"))
    
    out_dir_form = request.form.get("out_dir", "results").strip() or "results"
    base_out_path = (BASE_DIR / out_dir_form).resolve()

    try:
        base_out_path.relative_to(BASE_DIR.resolve())
    except ValueError:
        flash("out_dir 非法，必须位于项目目录内", "danger")
        return redirect(url_for("index"))

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_path = (base_out_path / run_tag).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # out_dir_form = request.form.get("out_dir", "results").strip() or "results"
    # out_path = (BASE_DIR / out_dir_form).resolve()

    # try:
    #     out_path.relative_to(BASE_DIR.resolve())
    # except ValueError:
    #     flash("out_dir 非法，必须位于项目目录内", "danger")
    #     return redirect(url_for("index"))

    out_path.mkdir(parents=True, exist_ok=True)
    save_users(users)

    args = SimpleNamespace(
        image=image,
        out_dir=str(out_path),
        num_users=num_users,
        code_length=code_length,
        colluders=colluders or ",".join(u["user_id"] for u in users[:num_users]),
        delta=delta,
        repeats=repeats,
        master_key=master_key,
        content_id=content_id,
        seed=seed,
        cover_size=cover_size,
        emit_psb=emit_psb,
        psb_max_side=psb_max_side,
        delta_mode=delta_mode,
        tile_size=tile_size,
    )

    try:
        route_info = apply_processing_mode_override(
            inspect_processing_mode(image),
            requested_mode=processing_mode,
            image_path=image,
        )
        if image and route_info["use_large_raster"]:
            report = demo_mod.run_large_demo(args)
            report["app_route"] = "large_raster_auto" if processing_mode == "auto" else "large_raster_forced"
            report["route_reason"] = route_info["reason"]
            report["processing_mode"] = processing_mode
            report["tile_size"] = tile_size
            write_unified_report(out_path, report=report)

            output_errors = []
            outputs = report.get("outputs", {}) if isinstance(report, dict) else {}
            for name, item in outputs.items():
                if name == "users":
                    continue
                if isinstance(item, dict) and item.get("psb_error"):
                    output_errors.append(f"{name}: {item['psb_error']}")

            user_outputs = outputs.get("users", {})
            if isinstance(user_outputs, dict):
                for user_id, item in user_outputs.items():
                    if isinstance(item, dict) and item.get("psb_error"):
                        output_errors.append(f"{user_id}: {item['psb_error']}")

            if output_errors:
                flash(
                    f"已切到大图分块链路（{route_info['reason']}），多用户/合谋结果已生成，但部分 PSB 导出失败："
                    + " | ".join(output_errors),
                    "warning",
                )
            else:
                flash(f"已切到大图分块链路（{route_info['reason']}），多用户水印、合谋攻击与评分已生成。", "success")
        
        else:
            report = demo_mod.run_demo(args)
            report["app_route"] = "default_auto" if processing_mode == "auto" else "default_forced"
            report["route_reason"] = route_info["reason"]
            report["processing_mode"] = processing_mode
            report["tile_size"] = tile_size
            write_unified_report(out_path, report=report)

            output_errors = []
            outputs = report.get("outputs", {}) if isinstance(report, dict) else {}
            for name, item in outputs.items():
                if name == "users":
                    continue
                if isinstance(item, dict) and item.get("psb_error"):
                    output_errors.append(f"{name}: {item['psb_error']}")

            user_outputs = outputs.get("users", {})
            if isinstance(user_outputs, dict):
                for user_id, item in user_outputs.items():
                    if isinstance(item, dict) and item.get("psb_error"):
                        output_errors.append(f"{user_id}: {item['psb_error']}")

            if output_errors:
                flash("运行完成，但部分 PSB 导出失败：" + " | ".join(output_errors), "warning")
            else:
                flash("水印嵌入并检测完成", "success")

    except Exception as exc:
        flash(f"运行失败: {exc}", "danger")

    return redirect(url_for("index", out_dir=str(out_path.relative_to(BASE_DIR))))


@app.route("/view/users")
def view_users():
    out_path = resolve_out_dir(request.args.get("out_dir"), default=RESULTS_DIR)
    files = sorted([p.name for p in out_path.iterdir() if p.is_file()]) if out_path.exists() else []
    report = load_report(out_path)

    if report and isinstance(report, dict) and report.get("user_results"):
        users = [{"user_id": uid, "user_key": ""} for uid in report["user_results"].keys()]
    else:
        users = load_users()

    return render_template(
        "users_view.html",
        users=users,
        images=files,
        out_dir=str(out_path.relative_to(BASE_DIR)),
        report=report,
        single_file_view=build_single_file_view(report, files),
    )


@app.route("/view/collusion")
def view_collusion():
    out_path = resolve_out_dir(request.args.get("out_dir"), default=RESULTS_DIR)
    files = sorted([p.name for p in out_path.iterdir() if p.is_file()]) if out_path.exists() else []
    report = load_report(out_path)
    return render_template(
        "collusion_view.html",
        images=files,
        out_dir=str(out_path.relative_to(BASE_DIR)),
        report=report,
        single_file_view=build_single_file_view(report, files),
    )


@app.route("/results/<path:filename>")
def results_file(filename: str):
    out_path = resolve_out_dir(request.args.get("out_dir"), default=RESULTS_DIR)
    return send_from_directory(str(out_path), filename)


@app.route("/report")
def view_report():
    out_path = resolve_out_dir(request.args.get("out_dir"), default=RESULTS_DIR)

    report_path = out_path / "report.json"
    if not report_path.exists():
        return render_template(
            "report.html",
            report_text=None,
            out_dir=str(out_path.relative_to(BASE_DIR)),
        )

    try:
        report_text = report_path.read_text(encoding="utf-8")
    except Exception:
        report_text = None

    return render_template(
        "report.html",
        report_text=report_text,
        out_dir=str(out_path.relative_to(BASE_DIR)),
    )


def load_report(out_path: Path) -> dict | None:
    report_path = out_path / "report.json"
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_display_text_from_image(
    image_path: str,
    master_key: str,
    content_id: str,
    delta: float,
    repeats: int,
    delta_mode: str,
    code_length: int,
    psb_max_side: int = 0,
    out_dir: str | Path | None = None,
    tile_size: int = 2048,
    route_mode: str = "auto",
) -> dict:
    route_info = apply_processing_mode_override(
        inspect_processing_mode(image_path),
        requested_mode=route_mode,
        image_path=image_path,
    )

    if route_info["use_large_raster"]:
        report = demo_mod.extract_large_single_file(
            image_path=image_path,
            out_dir=str(out_dir or RESULTS_DIR),
            content_id=content_id,
            master_key=master_key,
            code_length=code_length,
            delta=delta,
            repeats=repeats,
            delta_mode=delta_mode,
            tile_size=tile_size,
        )
        return {
            "image_path": image_path,
            "input_is_psb": bool(report.get("input_is_psb")),
            "input_depth": report.get("input_depth"),
            "display_bits_len": int(report.get("display_bits_len", demo_mod.DISPLAY_BITS_LEN)),
            "extracted_text": report.get("extracted_text", ""),
            "text_ok": bool(report.get("text_ok")),
            "route": "large_raster_auto" if route_mode == "auto" else "large_raster_forced",
            "route_reason": route_info["reason"],
        }

    image_bgr, source_meta = demo_mod.load_cover_and_meta(
        image_path=image_path,
        cover_size=512,
        psb_max_side=psb_max_side,
    )

    wm = demo_mod.BlindDwtDctQim(
        master_key=master_key,
        delta=delta,
        repeats=repeats,
        coeff_pos=(3, 3),
        bands=("LH", "HL"),
        delta_mode=delta_mode,
    )

    total_bits = demo_mod.DISPLAY_BITS_LEN + int(code_length)
    all_bits = wm.extract(
        image_bgr,
        n_bits=total_bits,
        content_id=content_id,
    )

    display_bits = all_bits[:demo_mod.DISPLAY_BITS_LEN]
    extracted_text, text_ok = demo_mod.decode_display_text(display_bits)

    return {
        "image_path": image_path,
        "input_is_psb": source_meta.input_is_psb,
        "input_depth": source_meta.depth,
        "display_bits_len": int(demo_mod.DISPLAY_BITS_LEN),
        "extracted_text": extracted_text,
        "text_ok": bool(text_ok),
        "route": "default_auto" if route_mode == "auto" else "default_forced",
        "route_reason": route_info["reason"],
    }


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
