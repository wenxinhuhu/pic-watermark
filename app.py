from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from tardos import UserRecord


BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "dev-secret-for-flask-session"


def ensure_data() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        # dump demo USER_DB as default
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
    # sync to demo module's USER_DB so the demo runner will use the updated list
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

    return render_template(
        "index.html",
        users=users,
        images=images,
        report=report,
        current_out_dir=str(out_path.relative_to(BASE_DIR)),
    )


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

    # 新增：PSB 相关参数
    emit_psb = request.form.get("emit_psb") == "on"
    psb_max_side = int(request.form.get("psb_max_side", 0))

    users = load_users()
    if num_users < 1 or num_users > len(users):
        flash(f"num_users 必须在 1 和 {len(users)} 之间", "danger")
        return redirect(url_for("index"))

    out_dir_form = request.form.get("out_dir", "results").strip() or "results"
    out_path = (BASE_DIR / out_dir_form).resolve()

    # 安全限制：禁止跳出项目目录
    try:
        out_path.relative_to(BASE_DIR.resolve())
    except ValueError:
        flash("out_dir 非法，必须位于项目目录内", "danger")
        return redirect(url_for("index"))

    out_path.mkdir(parents=True, exist_ok=True)

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
    )

    try:
        save_users(users)
        report = demo_mod.run_demo(args)

        output_errors = []
        outputs = report.get("outputs", {}) if isinstance(report, dict) else {}

        # 顶层输出：cover / pirate_average / pirate_xor
        for name, item in outputs.items():
            if name == "users":
                continue
            if isinstance(item, dict) and item.get("psb_error"):
                output_errors.append(f"{name}: {item['psb_error']}")

        # 用户级输出：user_1 / user_2 / ...
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
    users = load_users()
    report = load_report(out_path)
    return render_template(
        "users_view.html",
        users=users,
        images=files,
        out_dir=str(out_path.relative_to(BASE_DIR)),
        report=report,
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
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
