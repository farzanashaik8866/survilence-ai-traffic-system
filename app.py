"""
SURVILLENCE TRAFFIC — Unified AI Application
=============================================
Single unified entrypoint connecting:
  1. Frontend static files & single-page application routes
  2. Persistent user-isolated database (db.py)
  3. Real IISc UVH-26 YOLOv11-S model + ByteTrack vehicle tracking (detector.py)
  4. Complete REST API for Auth, Video Processing, History, Analytics, and Cameras

Runs locally via `python app.py` AND deploys directly to Vercel Serverless.
NO simulation. NO mock bounding boxes. 100% real AI model execution.
"""

import os
import sys
import json
import time
import uuid
import queue
import shutil
import threading
from urllib.parse import urlparse, urlunparse, quote
from pathlib import Path

from flask import Flask, request, jsonify, Response, send_file, send_from_directory, abort

import cv2
import numpy as np
from flask_cors import CORS

# Setup Python paths
ROOT_DIR = Path(__file__).resolve().parent
API_DIR = ROOT_DIR / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import db
from detector import process_video_analysis, MODEL_PT, MODEL_ONNX, get_yolo_detector

# Setup storage directories (writable in /tmp on Vercel)
TMP_DIR = Path("/tmp") if Path("/tmp").exists() else (ROOT_DIR / "storage")
UPLOAD_DIR = TMP_DIR / "traffic_uploads"
RESULTS_DIR = TMP_DIR / "traffic_results"
REPORTS_DIR = TMP_DIR / "traffic_reports"

for d in [UPLOAD_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Initialize Flask with frontend static serving
FRONTEND_DIR = ROOT_DIR / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app, origins=["*"])

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-User-Id"
    response.headers["Access-Control-Allow-Methods"] = "GET, PUT, POST, DELETE, OPTIONS"
    return response

# In-memory progress tracking for background jobs
_active_threads: dict[str, threading.Thread] = {}
_threads_lock = threading.Lock()

def get_current_user_id() -> str | None:
    raw = (
        request.headers.get("X-User-Id")
        or request.args.get("user_id")
        or request.form.get("user_id")
    )
    if not raw:
        return None
    cleaned = "".join(c for c in str(raw).strip() if c.isalnum() or c in ("_", "-"))
    return cleaned if cleaned else None

def safe_filename(name: str) -> str:
    name = Path(name).name
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ")
    return "".join(c if c in keep else "_" for c in name)[:100]

# ══════════════════════════════════════════════════════════════════════════════
# FRONTEND ROUTING
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index_page():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/login")
@app.route("/login.html")
def login_page():
    return send_from_directory(str(FRONTEND_DIR), "login.html")

@app.route("/register")
@app.route("/register.html")
def register_page():
    return send_from_directory(str(FRONTEND_DIR), "register.html")

# Stable aliases for every protected page. These remove the fragile dependency
# on one exact URL style (e.g. /pages/x.html vs /x.html vs /x).
PAGE_MAP = {
    "dashboard": "dashboard.html",
    "command-center": "command-center.html",
    "surveillance": "surveillance.html",
    "cameras": "cameras.html",
    "live": "live.html",
    "analytics": "analytics.html",
    "traffic-ai": "traffic-ai.html",
    "vehicle-search": "vehicle-search.html",
    "history": "history.html",
    "reports": "reports.html",
    "settings": "settings.html",
}

@app.route("/pages")
@app.route("/pages/")
def pages_root():
    return send_from_directory(str(FRONTEND_DIR / "pages"), "dashboard.html")

@app.route("/<page_name>.html")
def named_page_html(page_name: str):
    if page_name in PAGE_MAP:
        return send_from_directory(str(FRONTEND_DIR / "pages"), PAGE_MAP[page_name])
    if page_name in ("login", "register"):
        return send_from_directory(str(FRONTEND_DIR), f"{page_name}.html")
    abort(404)

@app.route("/<page_name>")
def named_page(page_name: str):
    if page_name in PAGE_MAP:
        return send_from_directory(str(FRONTEND_DIR / "pages"), PAGE_MAP[page_name])
    # Fallback to direct HTML file if exists
    target = FRONTEND_DIR / f"{page_name}.html"
    if target.exists():
        return send_from_directory(str(FRONTEND_DIR), f"{page_name}.html")
    abort(404)

@app.route("/pages/<path:filename>")
def pages_dir(filename: str):
    # Keep direct /pages/*.html URLs working for older bookmarks and links.
    return send_from_directory(str(FRONTEND_DIR / "pages"), filename)

@app.route("/css/<path:filename>")
def css_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "css"), filename)

@app.route("/js/<path:filename>")
def js_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "js"), filename)

@app.route("/assets/<path:filename>")
def assets_dir(filename: str):
    return send_from_directory(str(FRONTEND_DIR / "assets"), filename)

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
@app.route("/api/health")
def health():
    # Keep health checks fast: the heavy YOLO model is loaded lazily on the
    # first analysis/stream request instead of blocking application startup.
    return jsonify({
        "status": "ok",
        "app": "SURVILLENCE TRAFFIC AI",
        "mode": "Online Serverless / Direct App",
        "engine": "lazy-load",
        "model_loaded": bool(get_yolo_detector.__globals__.get("_cached_detector")),
        "model_pt_exists": MODEL_PT.exists(),
        "model_onnx_exists": MODEL_ONNX.exists(),
        "detector": "IISc UVH-26 YOLOv11-S + ByteTrack",
        "database": "Persistent Isolated Storage",
        "max_physical_videos_per_user": 1,
    })

# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/register", methods=["POST"])
@app.route("/api/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    name = data.get("fullName") or data.get("name")
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    user, error, code = db.create_user(name, username, email, password)
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "user": user}), code

@app.route("/api/auth/login", methods=["POST"])
@app.route("/api/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    identifier = data.get("identifier") or data.get("username") or data.get("email")
    password = data.get("password")

    user, error, code = db.authenticate_user(identifier, password)
    if error:
        return jsonify({"ok": False, "error": error, "code": "USER_NOT_FOUND" if code == 404 else "INVALID_PASSWORD"}), code
    return jsonify({"ok": True, "user": user}), code

@app.route("/api/auth/me", methods=["GET"])
@app.route("/api/me", methods=["GET"])
def auth_me():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"authenticated": False, "error": "Not authenticated"}), 401
    user = db.get_user(user_id)
    if not user:
        return jsonify({"authenticated": False, "error": "User does not exist"}), 401
    return jsonify({"authenticated": True, "user": user})

@app.route("/api/auth/settings", methods=["POST"])
def auth_settings():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    if db.update_user_settings(user_id, data):
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "User not found"}), 404

@app.route("/api/auth/password", methods=["POST"])
def auth_password():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    ok, error, code = db.change_password(user_id, data.get("currentPassword", ""), data.get("newPassword", ""))
    if not ok:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True})

@app.route("/api/auth/profile", methods=["POST"])
def auth_profile():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    user, error, code = db.update_user_profile(user_id, data.get("fullName"), data.get("email"))
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "user": user})

# ══════════════════════════════════════════════════════════════════════════════
# VIDEO ANALYSIS PIPELINE (REAL MODEL EXECUTION)
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisCancelled(Exception):
    pass

def _run_detection_worker(user_id: str, job_id: str, session_id: str, input_path: str, output_path: str, report_path: str, orig_name: str, analysis_settings: dict | None = None):
    try:
        def on_progress(pct, stage):
            db.update_job_progress(job_id, pct, stage)

        def cancel_check():
            return db.is_job_cancelled(job_id, user_id)

        report = process_video_analysis(
            input_path=input_path,
            output_video_path=output_path,
            output_report_path=report_path,
            session_id=session_id,
            progress_callback=on_progress,
            cancel_check=cancel_check,
            stride=int((analysis_settings or {}).get("frame_stride", 1)),
            max_dim=1280,
            settings=analysis_settings or {}
        )

        # A cancellation request may arrive just after the final frame.
        # Never resurrect a cancelled job with a completed history record.
        if cancel_check():
            raise AnalysisCancelled()

        # Save to database
        analysis_data = {
            "sessionId": session_id,
            "filename": orig_name,
            "status": "completed",
            "total_vehicles": report.get("total_vehicles", 0),
            "cars": report.get("cars", 0),
            "motorcycles": report.get("motorcycles", 0),
            "auto_rickshaws": report.get("auto_rickshaws", 0),
            "autos": report.get("auto_rickshaws", 0),
            "buses": report.get("buses", 0),
            "trucks": report.get("trucks", 0),
            "video_duration": report.get("video_duration", 0),
            "processing_time": report.get("processing_time_seconds", 0),
            "input_video": input_path,
            "output_video": output_path,
            "report_path": report_path,
        }
        db.add_analysis(user_id, analysis_data)
        db.save_report(user_id, session_id, report)
        # Persist newly detected intelligent incidents into the user's alert feed.
        for evt in report.get("incidents", [])[:100]:
            db.add_alert(
                user_id=user_id,
                alert_type=evt.get("type", "INCIDENT"),
                severity=evt.get("severity", "MEDIUM"),
                message=evt.get("message", "Traffic incident detected"),
                session_id=session_id,
                payload=evt,
            )
        db.complete_job(job_id, status="completed")

        # Clean up older physical video files for this user (Max 1 physical video retained)
        entries = db.get_user_history(user_id)
        for e in entries:
            sid = e.get("sessionId")
            if sid != session_id:
                for k in ("input_video", "output_video"):
                    p = e.get(k)
                    if p and Path(p).exists() and str(p) != output_path and str(p) != input_path:
                        try:
                            Path(p).unlink()
                        except Exception:
                            pass

    except AnalysisCancelled:
        print(f"[Detector Worker] Cancelled job {job_id}", flush=True)
        for path in (input_path, output_path, report_path):
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        for extra in (RESULTS_DIR.glob(f"raw_*{session_id}*"), RESULTS_DIR.glob(f"h264_*{session_id}*")):
            for path in extra:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
        db.complete_job(job_id, status="cancelled", error="Cancelled by user")
    except Exception as exc:
        print(f"[Detector Worker Error] {exc}", flush=True)
        db.complete_job(job_id, status="failed", error=str(exc))
        db.add_analysis(user_id, {
            "sessionId": session_id,
            "filename": orig_name,
            "status": "failed",
            "total_vehicles": 0,
            "input_video": input_path,
            "output_video": "",
            "report_path": "",
        })
    finally:
        with _threads_lock:
            _active_threads.pop(job_id, None)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized: user_id is required"}), 401

    # Enforce 1 active job per user
    existing_job = db.get_active_job(user_id)
    if existing_job:
        return jsonify({
            "error": "An analysis is already in progress. Please wait for it to complete.",
            "code": "ALREADY_PROCESSING",
            "activeJob": existing_job,
        }), 409

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{ext}'. Allowed: MP4, AVI, MOV, MKV, WebM"}), 400

    def _float_field(name, default):
        try: return float(request.form.get(name, default))
        except Exception: return default
    def _int_field(name, default):
        try: return int(request.form.get(name, default))
        except Exception: return default

    # User-level accuracy profile, with request-level overrides.
    user = db.get_user(user_id) or {}
    saved = user.get("settings", {}) or {}
    try: lane_count = int(request.form.get("lane_count", saved.get("lane_count", 3)))
    except Exception: lane_count = 3
    expected_direction = str(request.form.get("expected_direction", saved.get("expected_direction", "AUTO"))).upper()
    if expected_direction not in ("UP", "DOWN", "BOTH", "AUTO"): expected_direction = "AUTO"
    analysis_settings = {
        "expected_direction": expected_direction,
        "speed_limit_kmh": _float_field("speed_limit_kmh", saved.get("speed_limit_kmh", 40)),
        "meters_per_pixel": _float_field("meters_per_pixel", saved.get("meters_per_pixel", 0.05)),
        "lane_count": max(2, min(8, lane_count)),
        "stopped_seconds": _float_field("stopped_seconds", saved.get("stopped_seconds", 5)),
        "confidence": _float_field("confidence", saved.get("confidence", 0.30)),
        "frame_stride": max(1, min(4, _int_field("frame_stride", saved.get("frame_stride", 1)))),
        "image_size": max(416, min(960, _int_field("image_size", saved.get("image_size", 640)))),
        "nms_iou": _float_field("nms_iou", saved.get("nms_iou", 0.45)),
        "confirmation_frames": max(2, min(8, _int_field("confirmation_frames", saved.get("confirmation_frames", 3)))),
        "direction_window": max(5, min(20, _int_field("direction_window", saved.get("direction_window", 10)))),
        "direction_min_pixels": _float_field("direction_min_pixels", saved.get("direction_min_pixels", 10)),
        "speed_smoothing_window": max(3, min(10, _int_field("speed_smoothing_window", saved.get("speed_smoothing_window", 5)))),
        "speed_margin_kmh": _float_field("speed_margin_kmh", saved.get("speed_margin_kmh", 0)),
        "speed_mode": str(request.form.get("speed_mode", saved.get("speed_mode", "AUTO"))).upper(),
        "speed_confirmation_window": max(3, min(8, _int_field("speed_confirmation_window", saved.get("speed_confirmation_window", 5)))),
        "speed_confirmation_hits": max(2, min(8, _int_field("speed_confirmation_hits", saved.get("speed_confirmation_hits", 3)))),
        "lane_top_width_ratio": _float_field("lane_top_width_ratio", saved.get("lane_top_width_ratio", 0.60)),
        "lane_bottom_width_ratio": _float_field("lane_bottom_width_ratio", saved.get("lane_bottom_width_ratio", 1.0)),
        "speed_reference_distance_m": _float_field("speed_reference_distance_m", saved.get("speed_reference_distance_m", 0)),
        "speed_gate_top": _float_field("speed_gate_top", saved.get("speed_gate_top", 0.35)),
        "speed_gate_bottom": _float_field("speed_gate_bottom", saved.get("speed_gate_bottom", 0.70)),
    }

    if analysis_settings["speed_mode"] not in ("ESTIMATED", "GATES", "AUTO"):
        analysis_settings["speed_mode"] = "AUTO"

    session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    orig_name = safe_filename(f.filename)

    input_path = str(UPLOAD_DIR / f"{session_id}_{orig_name}")
    output_path = str(RESULTS_DIR / f"{session_id}_output.mp4")
    report_path = str(REPORTS_DIR / f"{session_id}_report.json")

    try:
        f.save(input_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to save video: {exc}"}), 500

    if not Path(input_path).exists() or Path(input_path).stat().st_size == 0:
        return jsonify({"error": "Uploaded video file is empty or missing"}), 400

    # Create server-side processing job
    job = db.create_job(user_id, session_id, orig_name)

    # Launch processing in background thread
    t = threading.Thread(
        target=_run_detection_worker,
        args=(user_id, job["jobId"], session_id, input_path, output_path, report_path, orig_name, analysis_settings),
        daemon=True,
    )
    with _threads_lock:
        _active_threads[job["jobId"]] = t
    t.start()

    return jsonify({
        "sessionId": session_id,
        "session_id": session_id,
        "jobId": job["jobId"],
        "status": "processing",
        "filename": orig_name,
        "stage": "Upload received and validated",
        "userId": user_id,
        "settings": analysis_settings,
    })

@app.route("/api/active-job")
def active_job():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"job": None})
    job = db.get_active_job(user_id)
    return jsonify({"job": job, "active": bool(job)})

@app.route("/api/completed-job")
def completed_job():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"job": None})
    job = db.get_recently_completed_job(user_id)
    return jsonify({"job": job})

@app.route("/api/completed-job/<session_id>/dismiss", methods=["POST"])
def dismiss_completed_job(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False}), 401
    db.dismiss_completed_job(user_id, session_id)
    return jsonify({"ok": True})

@app.route("/api/job/<job_id>")
def get_job_status(job_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    job = db.get_active_job(user_id)
    if job and job.get("jobId") == job_id:
        return jsonify(job)
    recent = db.get_recently_completed_job(user_id)
    if recent and recent.get("jobId") == job_id:
        return jsonify(recent)
    return jsonify({"status": "unknown"})

@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    job = db.get_active_job(user_id)
    if job and job.get("jobId") == job_id:
        db.complete_job(job_id, status="cancelled", error="Cancelled by user")
        return jsonify({"ok": True, "status": "cancelled"})
    return jsonify({"ok": False, "error": "Job not found or not active"}), 404

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS, REPORTS & DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/result/<session_id>")
def result(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        return jsonify({"error": "Analysis not found or access denied"}), 404

    report_data = db.get_report(user_id, session_id)
    if not report_data and analysis.get("report_path"):
        rp = Path(analysis["report_path"])
        if rp.exists():
            try:
                report_data = json.loads(rp.read_text("utf-8"))
            except Exception:
                pass

    return jsonify({
        "status": analysis["status"],
        "report": report_data,
        "sessionId": session_id,
        "userId": user_id,
        "videoAvailable": analysis.get("videoAvailable", False),
        "historyEntry": analysis,
    })

@app.route("/api/video/<session_id>")
@app.route("/api/stream/video/<session_id>")
def video(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        abort(401)

    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        abort(404)

    out_path = analysis.get("output_video")
    if out_path and Path(out_path).exists():
        return send_file(str(out_path), mimetype="video/mp4", conditional=True)

    candidates = list(RESULTS_DIR.glob(f"{session_id}*.*"))
    if candidates:
        return send_file(str(candidates[0]), mimetype="video/mp4", conditional=True)

    abort(404)

@app.route("/api/download/video/<session_id>")
def download_video(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        abort(401)
    analysis = db.get_user_analysis(user_id, session_id)
    if not analysis:
        abort(404)
    out_path = analysis.get("output_video")
    if out_path and Path(out_path).exists():
        return send_file(str(out_path), as_attachment=True, download_name=f"tracked_{session_id}.mp4")
    abort(404)

@app.route("/api/download/report/<session_id>")
def download_report(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        abort(401)
    report_data = db.get_report(user_id, session_id)
    if not report_data:
        analysis = db.get_user_analysis(user_id, session_id)
        if analysis and analysis.get("report_path") and Path(analysis["report_path"]).exists():
            report_data = json.loads(Path(analysis["report_path"]).read_text("utf-8"))
    if report_data:
        return Response(
            json.dumps(report_data, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=report_{session_id}.json"}
        )
    abort(404)

# ══════════════════════════════════════════════════════════════════════════════
# HISTORY & STORAGE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/history")
def history():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify([])
    entries = db.get_user_history(user_id)
    return jsonify(entries)

@app.route("/api/history/<session_id>", methods=["DELETE"])
def delete_history(session_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    ok = db.delete_user_analysis(user_id, session_id)
    if not ok:
        return jsonify({"error": "Not found or access denied"}), 404
    return jsonify({"status": "deleted", "sessionId": session_id})

@app.route("/api/storage/stats")
def storage_stats():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({
            "videos_in_bytes": 0,
            "videos_out_bytes": 0,
            "reports_bytes": 0,
            "total_bytes": 0,
            "history_count": 0,
            "retained_videos": 0,
        })
    entries = db.get_user_history(user_id)
    retained = sum(1 for e in entries if e.get("videoAvailable", False))

    in_size = sum(Path(e["input_video"]).stat().st_size for e in entries if e.get("input_video") and Path(e["input_video"]).exists())
    out_size = sum(Path(e["output_video"]).stat().st_size for e in entries if e.get("output_video") and Path(e["output_video"]).exists())
    rep_size = sum(Path(e["report_path"]).stat().st_size for e in entries if e.get("report_path") and Path(e["report_path"]).exists())

    return jsonify({
        "videos_in_bytes": in_size,
        "videos_out_bytes": out_size,
        "reports_bytes": rep_size,
        "total_bytes": in_size + out_size + rep_size,
        "history_count": len(entries),
        "retained_videos": retained,
    })

@app.route("/api/storage/cleanup", methods=["POST"])
def storage_cleanup():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"removed_files": 0})
    entries = db.get_user_history(user_id)
    retained_outputs = {e.get("output_video") for e in entries if e.get("videoAvailable", False)}
    removed = 0

    for e in entries:
        if not e.get("videoAvailable", False):
            for k in ("input_video", "output_video"):
                p = e.get(k)
                if p and p not in retained_outputs and Path(p).exists():
                    try:
                        Path(p).unlink()
                        removed += 1
                    except Exception:
                        pass
    return jsonify({"removed_files": removed})

# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE: ALERTS, VEHICLE SEARCH & BUILT-IN AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/alerts")
def alerts_list():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify([])
    try: limit = int(request.args.get("limit", 100))
    except Exception: limit = 100
    unread_only = request.args.get("unread", "0") == "1"
    return jsonify(db.get_alerts(user_id, limit=limit, include_ack=not unread_only))

@app.route("/api/alerts/<alert_id>/ack", methods=["POST"])
def alert_ack(alert_id):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    return jsonify({"ok": db.acknowledge_alert(user_id, alert_id)})

@app.route("/api/vehicle-search")
def vehicle_search():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify([])
    results = db.search_vehicles(
        user_id=user_id,
        query=request.args.get("q", ""),
        vehicle_type=request.args.get("type", ""),
        direction=request.args.get("direction", ""),
        lane=request.args.get("lane", ""),
        violation=request.args.get("violation", ""),
    )
    return jsonify(results)

CHAT_KB = [
    ("bytetrack", "ByteTrack keeps a track ID across frames. This lets the system count unique vehicles and build a trajectory instead of counting the same vehicle repeatedly."),
    ("yolov11", "YOLO detects vehicles in each frame. ByteTrack then associates those detections over time; detection and tracking are separate stages."),
    ("wrong way", "Wrong-way detection uses the bottom-center trajectory, a direction vote window, and multiple consistent observations before raising an alert. Set the expected road direction correctly for your camera."),
    ("speed", "Speed is smoothed over several trajectory samples. For more reliable physical km/h, configure two virtual speed gates with a known distance; pixel-scale speed is only an estimate."),
    ("lane", "Lane analysis uses the vehicle's bottom-center point and a perspective-aware trapezoid. Tune the top/bottom road width in Detection Settings when the camera is strongly angled."),
    ("incident", "The intelligence layer looks for persistent wrong-way movement, persistent speed violations, prolonged stopped vehicles, and sudden/high congestion. One-frame noise is filtered out."),
    ("traffic density", "Traffic density is derived from simultaneously active confirmed tracks, not raw detections. LOW/MEDIUM/HIGH/SEVERE thresholds are operational heuristics."),
    ("vehicle search", "Vehicle Search indexes completed reports by track ID, vehicle class, direction, lane, and incident flags."),
    ("camera", "Multi-camera monitoring supports saved CCTV/IP/Webcam sources. The Flask machine must be able to reach private RTSP streams."),
    ("no api", "The Traffic Assistant is local and rule-based; it does not call an external AI service. The bundled detector also runs locally on the machine."),
    ("accuracy", "No software can honestly claim a fixed detection accuracy without a labeled test set. This project improves reliability using higher-resolution inference, temporal confirmation, class voting, robust motion smoothing, and perspective-aware lanes."),
    ("calibration", "For speed, measure a real distance on the road and set the two virtual gate lines plus that distance. The resulting gate-crossing speed is more stable than raw pixel velocity."),
]


def _latest_report_for_user(user_id, session_id=""):
    if session_id:
        return db.get_report(user_id, session_id), session_id
    history = db.get_user_history(user_id)
    for row in history:
        sid = row.get("sessionId") or row.get("session_id")
        if sid and row.get("status") == "completed":
            rep = db.get_report(user_id, sid)
            if rep:
                return rep, sid
    return None, None


def _chat_answer(message: str, context: dict | None = None):
    raw = (message or "").strip()
    text = raw.lower()
    if not text:
        return {"answer": "Ask me about this analysis, accuracy, YOLO, ByteTrack, speed, wrong-way detection, lanes, incidents, cameras, or vehicle search.", "suggestions": ["Summarize this analysis", "How can I improve accuracy?", "Why is speed estimated?"], "action": None}

    # Navigation/action intents make the assistant operational, not just explanatory.
    actions = {
        "vehicle search": ("Open Vehicle Search", "/vehicle-search"),
        "search vehicle": ("Open Vehicle Search", "/vehicle-search"),
        "live monitoring": ("Open Live Monitoring", "/live"),
        "multi camera": ("Open Live Monitoring", "/live"),
        "cameras": ("Open CCTV Connectivity", "/cameras"),
        "camera": ("Open CCTV Connectivity", "/cameras"),
        "settings": ("Open Detection Settings", "/settings"),
        "accuracy settings": ("Open Detection Settings", "/settings"),
        "new analysis": ("Start New Analysis", "/surveillance"),
    }
    for key, (label, url) in actions.items():
        if key in text and any(x in text for x in ("open", "go", "show", "take me", "where")):
            return {"answer": f"I can take you there: {label}.", "suggestions": [], "action": {"label": label, "url": url}}

    if context:
        v = context.get("vehicle_counts") or context.get("vehicles") or {}
        total = int(context.get("total_vehicles", context.get("total_unique", 0)) or 0)
        inc = context.get("incident_summary", {}) or {}
        wrong = int(inc.get("wrong_way", context.get("wrong_way", {}).get("count", 0)) or 0)
        speed = int(inc.get("speed_violation", context.get("speed_violations", {}).get("count", 0)) or 0)
        stopped = int(inc.get("stopped_vehicle", 0) or 0)
        density = (context.get("traffic_density") or {}).get("level", "N/A")
        quality = context.get("analysis_quality") or {}
        lane = (context.get("lane_analysis") or {}).get("lanes", {})
        top_lane = max(lane.items(), key=lambda kv: (kv[1].get("average_active", 0), kv[1].get("peak_active", 0)))[0] if lane else None

        if any(k in text for k in ("summarize", "summary", "overview", "what happened")):
            bits = [f"This session confirmed {total} unique vehicles and ended at {density} traffic density."]
            if inc.get("total", 0): bits.append(f"I found {inc.get('total')} intelligent incident(s): {wrong} wrong-way, {speed} speed, and {stopped} prolonged-stop signal(s).")
            if top_lane: bits.append(f"{top_lane} carried the highest sustained active load.")
            if quality: bits.append(f"Internal analysis quality is {quality.get('score', '—')}/100 ({quality.get('level', '—')}); this is not a measured model accuracy score.")
            return {"answer": " ".join(bits), "suggestions": ["Which lane is most congested?", "Explain the incidents", "How do I improve accuracy?"], "action": {"label": "Open Traffic AI", "url": f"/traffic-ai?session={context.get('session_id','')}"} if context.get("session_id") else None}
        if "lane" in text and any(k in text for k in ("most", "busy", "congested", "highest")):
            if not top_lane: return {"answer": "Lane statistics are not available in this report.", "suggestions": [], "action": None}
            ls = lane[top_lane]
            return {"answer": f"{top_lane} is the busiest lane in this session, with about {ls.get('average_active',0):.1f} active vehicles on average and a peak of {ls.get('peak_active',0)}. The lane model is perspective-aware, but camera geometry still affects assignment quality.", "suggestions": ["How can I improve lane accuracy?"], "action": None}
        if any(k in text for k in ("vehicle #", "track #", "track id", "vehicle id")):
            import re
            m = re.search(r"(?:#|id\s*)(\d+)", text)
            if m:
                tid = int(m.group(1))
                match = next((x for x in (context.get("vehicles_detail") or []) if int(x.get("track_id", -1)) == tid), None)
                if match:
                    flags=[]
                    if match.get("wrong_way"): flags.append("wrong-way")
                    if match.get("speed_violation"): flags.append("speed violation")
                    flag_text = ("; " + ", ".join(flags)) if flags else ""
                    return {"answer": f"Vehicle #{tid}: {match.get('type', match.get('class','vehicle'))}, Lane {match.get('lane','—')}, direction {match.get('direction','UNKNOWN')}, speed {match.get('speed_kmh',0):.1f} km/h{flag_text}.", "suggestions": ["Explain this vehicle", "Show the vehicle in Vehicle Search"], "action": {"label":"Open Vehicle Search", "url":f"/vehicle-search?q={tid}"}}
                return {"answer": f"I could not find track #{tid} in the selected report. Track IDs are local to an analysis session, so check the session selector first.", "suggestions": ["Summarize this analysis", "Open Vehicle Search"], "action": {"label":"Open Vehicle Search", "url":f"/vehicle-search?q={tid}"}}
        if any(k in text for k in ("incident", "violation", "wrong way", "speeding")):
            examples = (context.get("incidents") or [])[:3]
            extra = "" if not examples else " Recent examples: " + "; ".join(e.get("message", e.get("type", "incident")) for e in examples)
            return {"answer": f"This report contains {inc.get('total',0)} intelligent incident(s): {wrong} wrong-way, {speed} speed, {stopped} stopped-vehicle. Alerts require temporal evidence before being raised to reduce one-frame noise.{extra}", "suggestions": ["Why did this alert trigger?", "How does wrong-way detection work?", "How is speed calculated?"], "action": None}
        if any(k in text for k in ("why did", "why was", "trigger", "alert trigger")) and inc.get('total',0):
            first = (context.get("incidents") or [None])[0]
            if first:
                return {"answer": f"The latest incident was {first.get('type','INCIDENT')}: {first.get('message','No detail recorded')}. The detector waits for temporal/trajectory evidence rather than treating a single frame as a final alert.", "suggestions": ["Explain analysis quality", "Open Traffic AI"], "action": {"label":"Inspect Traffic AI", "url":f"/traffic-ai?session={context.get('session_id','')}"} if context.get('session_id') else None}
        if any(k in text for k in ("quality", "confidence score", "analysis score", "reliability")):
            return {"answer": f"This run's internal analysis-quality signal is {quality.get('score','—')}/100 ({quality.get('level','—')}). It combines detection confidence, confirmed-track ratio and track stability; it is a consistency indicator, not a benchmark precision/recall score.", "suggestions": ["How can I improve accuracy?", "Open accuracy settings"], "action": {"label":"Open Detection Settings", "url":"/settings"}}
        if any(k in text for k in ("what should i do", "what do i do", "next step", "recommendation", "recommendations")):
            recs = context.get("recommendations") or ["Validate the camera geometry and review flagged incidents before operational use."]
            answer = "Recommended checks: " + " ".join(f"{i+1}) {r}" for i,r in enumerate(recs[:4]))
            return {"answer": answer, "suggestions": ["Explain analysis quality", "Open Command Center"], "action": {"label":"Open Command Center", "url":"/command-center"}}
        if any(k in text for k in ("improve", "better", "accurate", "accuracy", "false positive", "false negative")):
            return {"answer": "For better reliability, use frame stride 1, confidence around 0.25–0.35, 640–960px inference, correct expected direction, tune the lane trapezoid, and calibrate two real-distance speed gates. Most importantly, validate against a small labeled sample from your own cameras; that is how you measure real precision/recall.", "suggestions": ["Open accuracy settings", "How do speed gates work?"], "action": {"label": "Open Detection Settings", "url": "/settings"}}
        if "how many" in text and "vehicle" in text:
            return {"answer": f"This session has {total} confirmed unique vehicles. Cars: {v.get('car',0)}, motorcycles: {v.get('motorcycle',0)}, autos: {v.get('auto',v.get('auto_rickshaw',0))}, buses: {v.get('bus',0)}, trucks: {v.get('truck',0)}.", "suggestions": ["Summarize this analysis"], "action": None}

    for phrase, answer in CHAT_KB:
        tokens = phrase.split()
        if all(token in text for token in tokens):
            return {"answer": answer, "suggestions": ["How can I improve accuracy?", "Summarize this analysis"], "action": None}
    if text in {"hi", "hello", "hey", "help"}:
        return {"answer": "Hi! I can explain the traffic pipeline, interpret the current report, help troubleshoot accuracy, or take you to Vehicle Search, Live Monitoring, CCTV, Settings, or a new analysis.", "suggestions": ["Summarize this analysis", "How can I improve accuracy?", "Open Vehicle Search"], "action": None}
    return {"answer": "I can help with project doubts and live analysis interpretation. Try asking: ‘summarize this analysis’, ‘which lane is busiest?’, ‘why is speed estimated?’, ‘how does wrong-way detection work?’, or ‘how can I improve accuracy?’", "suggestions": ["Summarize this analysis", "How can I improve accuracy?", "Which lane is busiest?"], "action": None}


@app.route("/api/chat", methods=["POST"])
def chat():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    sid = data.get("sessionId", "") or ""
    context, resolved_sid = _latest_report_for_user(user_id, sid)
    if context:
        context = dict(context)
        context["session_id"] = resolved_sid
    result = _chat_answer(message, context=context)
    return jsonify({"ok": True, "answer": result["answer"], "suggestions": result.get("suggestions", []), "action": result.get("action"), "mode": "local-project-assistant"})


@app.route("/api/intelligence/overview")
def intelligence_overview():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    report, sid = _latest_report_for_user(user_id)
    cameras = db.get_cameras(user_id)
    alerts = db.get_alerts(user_id, limit=40)
    if not report:
        return jsonify({"ok": True, "session": None, "alerts": alerts, "cameras": {"total": len(cameras), "online": sum(1 for c in cameras if c.get("status") == "ONLINE")}})
    lanes = (report.get("lane_analysis") or {}).get("lanes", {})
    top_lane = max(lanes.items(), key=lambda kv: (kv[1].get("average_active",0), kv[1].get("peak_active",0)))[0] if lanes else None
    inc = report.get("incident_summary") or {}
    quality = report.get("analysis_quality") or {}
    safety_score = 100
    safety_score -= min(40, int(inc.get("wrong_way",0))*10)
    safety_score -= min(30, int(inc.get("speed_violation",0))*6)
    safety_score -= min(20, int(inc.get("stopped_vehicle",0))*3)
    safety_score -= min(20, int(inc.get("high_congestion",0))*5 + int(inc.get("congestion_spike",0))*4)
    safety_score = max(0, safety_score)
    return jsonify({
        "ok": True,
        "session": {"id": sid, "report": report},
        "summary": {
            "safety_score": safety_score,
            "analysis_quality": quality,
            "busiest_lane": top_lane,
            "traffic_level": (report.get("traffic_density") or {}).get("level", "N/A"),
            "total_vehicles": report.get("total_vehicles", 0),
            "incident_total": inc.get("total", 0),
            "recommendations": report.get("recommendations", []),
        },
        "alerts": alerts,
        "cameras": {"total": len(cameras), "online": sum(1 for c in cameras if c.get("status") == "ONLINE")},
    })

# ══════════════════════════════════════════════════════════════════════════════
# CAMERAS & CCTV CONNECTIVITY
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/cameras", methods=["GET"])
def cameras_list():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify([]), 200
    cameras = db.get_cameras(user_id)
    return jsonify(cameras)

@app.route("/api/cameras", methods=["POST"])
def cameras_create():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    name = data.get("name") or data.get("cameraName")
    camera_type = data.get("type") or data.get("cameraType") or "IP Camera"
    stream_url = data.get("streamUrl") or data.get("url") or ""
    username = data.get("username")
    password = data.get("password")
    port = data.get("port")

    cam, error, code = db.create_camera(user_id, name, camera_type, stream_url, username, password, port)
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "camera": cam}), code

@app.route("/api/cameras/<camera_id>", methods=["GET"])
def cameras_get(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    cam = db.get_camera(user_id, camera_id)
    if not cam:
        return jsonify({"error": "Camera not found or access denied"}), 404
    return jsonify(cam)

@app.route("/api/cameras/<camera_id>", methods=["PUT", "POST"])
def cameras_update(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    cam, error, code = db.update_camera(user_id, camera_id, data)
    if error:
        return jsonify({"ok": False, "error": error}), code
    return jsonify({"ok": True, "camera": cam})

@app.route("/api/cameras/<camera_id>", methods=["DELETE"])
def cameras_delete(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    ok = db.delete_camera(user_id, camera_id)
    if not ok:
        return jsonify({"ok": False, "error": "Camera not found or unauthorized"}), 404
    return jsonify({"ok": True, "deleted": camera_id})

def _camera_runtime_url(cam: dict) -> str:
    """Build a usable capture URL, including optional stored camera credentials."""
    stream_url = str(cam.get("streamUrl") or cam.get("stream_url") or "").strip()
    if not stream_url:
        return ""
    if str(cam.get("type") or cam.get("camera_type") or "").strip().lower() == "webcam" and stream_url.isdigit():
        return stream_url
    try:
        parsed = urlparse(stream_url)
        username = str(cam.get("username") or "").strip()
        password = str(cam.get("password") or "")
        if parsed.scheme.lower() in {"rtsp", "http", "https"} and parsed.hostname and username and not parsed.username:
            host = parsed.hostname
            if ':' in host and not host.startswith('['):
                host = f"[{host}]"
            netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{host}"
            if parsed.port:
                netloc += f":{parsed.port}"
            stream_url = urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    except Exception:
        pass
    return stream_url

def _probe_camera_connection(stream_url: str, camera_type: str = "", username: str = "", password: str = ""):
    stream_url = (stream_url or "").strip()
    camera_type = (camera_type or "").strip().lower()

    # Local webcam/device index (0, 1, ...).
    if camera_type == "webcam" or (stream_url.isdigit() and camera_type in {"webcam", "local camera", "other supported source"}):
        try:
            cap = cv2.VideoCapture(int(stream_url or "0"))
            if not cap.isOpened():
                return "OFFLINE", None, None, f"Webcam device {stream_url or '0'} could not be opened."
            ret, frame = cap.read()
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                return "ONLINE", f"{w} × {h}", fps, None
            return "NO SIGNAL", None, fps, "Webcam opened but returned no frame."
        except Exception as exc:
            return "CONNECTION ERROR", None, None, str(exc)

    if not stream_url:
        return "NOT CONFIGURED", None, None, "No stream URL provided."

    stream_url = _camera_runtime_url({"streamUrl": stream_url, "type": camera_type, "username": username, "password": password})
    parsed = urlparse(stream_url)
    host = (parsed.hostname or "").lower()
    is_private = (host == "localhost" or host == "127.0.0.1" or host.startswith("192.168.") or host.startswith("10.") or
                  (host.startswith("172.") and len(host.split('.')) == 4 and 16 <= int(host.split('.')[1]) <= 31))
    if is_private and os.environ.get("VERCEL") == "1":
        return (
            "PRIVATE LAN / UNREACHABLE",
            None,
            None,
            "This camera address is on a private local network. A cloud-deployed server cannot reach it without a secure routed endpoint."
        )

    if not (parsed.scheme.lower() in {"rtsp", "http", "https"}):
        return "UNSUPPORTED STREAM", None, None, "Stream URL must begin with rtsp://, http://, or https://"

    try:
        import cv2
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            return "OFFLINE", None, None, "Camera stream could not be opened from cloud."
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return "ONLINE", f"{w} × {h}", 25.0, None
        return "NO SIGNAL", None, None, "Stream opened but no frames received."
    except Exception as exc:
        return "CONNECTION ERROR", None, None, str(exc)

@app.route("/api/cameras/test", methods=["POST"])
def cameras_test_raw():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    stream_url = data.get("streamUrl") or data.get("url") or ""
    status, resolution, fps, error = _probe_camera_connection(stream_url, data.get("type") or data.get("cameraType") or "", data.get("username") or "", data.get("password") or "")
    return jsonify({
        "status": status,
        "online": status == "ONLINE",
        "resolution": resolution,
        "fps": fps,
        "error": error,
    })

@app.route("/api/cameras/<camera_id>/test", methods=["POST"])
def cameras_test_saved(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    cam = db.get_camera(user_id, camera_id, include_password=True)
    if not cam:
        return jsonify({"ok": False, "error": "Camera not found or access denied"}), 404

    status, resolution, fps, error = _probe_camera_connection(cam.get("streamUrl"), cam.get("type"), cam.get("username") or "", cam.get("password") or "")
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    updates = {"status": status}
    if status == "ONLINE":
        updates["lastConnected"] = now_iso
        if resolution:
            updates["resolution"] = resolution
        if fps:
            updates["fps"] = fps

    updated_cam, _, _ = db.update_camera(user_id, camera_id, updates)
    return jsonify({
        "status": status,
        "online": status == "ONLINE",
        "resolution": resolution,
        "fps": fps,
        "error": error,
        "camera": updated_cam,
    })

@app.route("/api/cameras/<camera_id>/stats")
def camera_live_stats(camera_id: str):
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401
    cam = db.get_camera(user_id, camera_id, include_password=True)
    if not cam:
        return jsonify({"error": "Camera not found"}), 404

    counts = {"car":0,"motorcycle":0,"auto":0,"bus":0,"truck":0}
    active = 0
    status = cam.get("status", "NOT CONFIGURED")
    try:
        cap = _camera_capture(cam)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                det = get_yolo_detector()
                if det:
                    h,w=frame.shape[:2]
                    scale=min(1.0,640/max(h,w))
                    if scale < 1: frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
                    from detector import run_inference_on_frame
                    detections=run_inference_on_frame(det,frame,conf_thresh=0.30,imgsz=512)
                    active=len(detections)
                    for d in detections:
                        counts[d.get("class","car")]=counts.get(d.get("class","car"),0)+1
                    status="ONLINE"
        else:
            status="OFFLINE"
    except Exception:
        status = "OFFLINE"

    return jsonify({
        "cameraId": camera_id,
        "cameraName": cam["name"],
        "cameraType": cam["type"],
        "stats": {
            "active_vehicles": active,
            "cars": counts["car"],
            "motorcycles": counts["motorcycle"],
            "autos": counts["auto"],
            "buses": counts["bus"],
            "trucks": counts["truck"],
            "fps": cam.get("fps") or 25.0,
            "status": status,
            "last_sampled": time.strftime("%H:%M:%S"),
        }
    })

# ── Multi-camera MJPEG stream proxy with lightweight live detection ───────────
def _camera_capture(cam):
    url = _camera_runtime_url(cam)
    if str(cam.get("type", "")).lower() == "webcam" and str(url).strip().isdigit():
        return cv2.VideoCapture(int(str(url).strip()))
    return cv2.VideoCapture(url)


def _mjpeg_generator(user_id: str, camera_id: str):
    cam = db.get_camera(user_id, camera_id, include_password=True)
    if not cam:
        return
    cap = _camera_capture(cam)
    if not cap.isOpened():
        return
    from detector import SimpleByteTrack, run_inference_on_frame, get_yolo_detector, COLORS, DISPLAY_LABELS
    detector = get_yolo_detector()
    tracker = SimpleByteTrack(high_thresh=0.35, low_thresh=0.15)
    frame_no = 0
    seen_events=set()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    # Live camera streams do not have a full-clip lane-flow learning phase, so a
    # default "UP = wrong way" rule is unsafe. Cameras can opt in explicitly via
    # camera.expected_direction = UP/DOWN. Speed alerts use the camera calibration
    # when present and otherwise use the standard 0.05 m/px estimate.
    live_expected_direction = str(cam.get("expected_direction", "NONE")).upper()
    if live_expected_direction not in ("UP", "DOWN"):
        live_expected_direction = "NONE"
    live_speed_limit = float(cam.get("speed_limit_kmh") or 40.0)
    live_mpp = float(cam.get("meters_per_pixel") or 0.05)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_no += 1
            if frame_no % 2 == 0 and detector:
                dets = run_inference_on_frame(detector, frame, conf_thresh=0.25, imgsz=512)
                last_tracks = tracker.update(dets, frame_index=frame_no)
            tracks = last_tracks
            cv2.rectangle(frame, (8, 8), (340, 58), (8, 12, 18), -1)
            cv2.putText(frame, cam.get("name", camera_id)[:28], (18, 29), cv2.FONT_HERSHEY_SIMPLEX, .55, (235,242,250), 1)
            cv2.putText(frame, f"LIVE  |  {len(tracks)} vehicles", (18, 49), cv2.FONT_HERSHEY_SIMPLEX, .42, (110,220,180), 1)
            live_event_count=0
            for t in tracks:
                box=[int(v) for v in t["box"]]
                cls=t["class"]
                hist=t.get("history",[])
                direction="UNKNOWN"; speed=0.0
                if len(hist)>=2:
                    a,b=hist[max(0,len(hist)-8)],hist[-1]
                    dy=b[1]-a[1]
                    if abs(dy)>=4: direction="DOWN" if dy>0 else "UP"
                    if a[2] is not None and b[2] is not None and b[2]>a[2]:
                        dt=(b[2]-a[2])/max(fps,1.0)
                        if dt>0: speed=(float(np.hypot(b[0]-a[0],b[1]-a[1]))/dt)*live_mpp*3.6
                wrong_way=(live_expected_direction in ("UP", "DOWN") and direction in ("UP", "DOWN") and direction != live_expected_direction)
                speeding=(live_speed_limit > 0 and live_mpp > 0 and speed>=live_speed_limit and speed>0)
                if wrong_way and (t["track_id"],"WRONG_WAY") not in seen_events:
                    seen_events.add((t["track_id"],"WRONG_WAY")); live_event_count+=1
                    db.add_alert(user_id,"WRONG_WAY",f"{cam.get('name',camera_id)}: vehicle #{t['track_id']} moving the opposite direction","HIGH",camera_id=camera_id,payload={"track_id":t["track_id"]})
                if speeding and (t["track_id"],"SPEED_VIOLATION") not in seen_events:
                    seen_events.add((t["track_id"],"SPEED_VIOLATION")); live_event_count+=1
                    db.add_alert(user_id,"SPEED_VIOLATION",f"{cam.get('name',camera_id)}: vehicle #{t['track_id']} estimated at {speed:.1f} km/h","HIGH",camera_id=camera_id,payload={"track_id":t["track_id"],"speed_kmh":round(speed,2)})
                color=(40,80,255) if wrong_way or speeding else COLORS.get(cls,(200,200,200))
                x1,y1,x2,y2=box
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                badge=f"#{t['track_id']} {DISPLAY_LABELS.get(cls,cls)}"
                if speed>0: badge += f" {speed:.0f}km/h"
                if wrong_way: badge += " WRONG"
                elif speeding: badge += " SPEED"
                cv2.putText(frame,badge,(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            if live_event_count:
                cv2.putText(frame,f"ALERTS +{live_event_count}",(18,78),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,100,255),2)
            ok, encoded=cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok: continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(encoded)).encode() + b'\r\n\r\n' + encoded.tobytes() + b'\r\n'
    finally:
        cap.release()

@app.route("/api/cameras/<camera_id>/stream")
def camera_stream(camera_id: str):
    user_id=get_current_user_id()
    if not user_id:
        abort(401)
    cam=db.get_camera(user_id,camera_id, include_password=True)
    if not cam:
        abort(404)
    return Response(_mjpeg_generator(user_id,camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/cameras/<camera_id>/snapshot")
def camera_snapshot(camera_id: str):
    user_id=get_current_user_id()
    if not user_id: abort(401)
    cam=db.get_camera(user_id,camera_id, include_password=True)
    if not cam: abort(404)
    cap=_camera_capture(cam)
    if not cap.isOpened(): abort(503)
    ok, frame=cap.read(); cap.release()
    if not ok: abort(503)
    ok, encoded=cv2.imencode('.jpg',frame)
    if not ok: abort(500)
    return Response(encoded.tobytes(), mimetype='image/jpeg')

# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(f"SURVILLENCE TRAFFIC AI — Running on http://localhost:{port}")
    print(f"Model PT:   {MODEL_PT} (exists: {MODEL_PT.exists()})")
    print(f"Model ONNX: {MODEL_ONNX} (exists: {MODEL_ONNX.exists()})")
    print("AI model: lazy loading enabled — startup will not wait for YOLO")
    print("Open http://localhost:5000 in your browser.")
    print("Page aliases are available at /dashboard, /surveillance, /cameras, /live, /analytics, /traffic-ai, /vehicle-search, /history, /reports, /settings.")
    print("The first video/live-camera analysis may take a little longer while the model loads.")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
