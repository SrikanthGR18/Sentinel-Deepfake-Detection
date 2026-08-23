from flask import Flask, render_template, request, redirect, session, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import subprocess
import uuid
import shutil

from ammp import is_audio_only_upload, preprocess_media
from svim import analyze_audio
from temporal import analyze_temporal
from fusion import FusionClassification, FusionInput, dashboard_result_labels, fuse_trust_scores
from models.inference import predict_video

# =========================================================
# Sentinel-AMMP Framework
# Adaptive Multimodal Micro-Artifact Profiling
#
# Modules:
# AFCP  -> Adaptive Frame Consistency Profiling
# SVIM  -> Spectral Voice Irregularity Mapping
# DTFE  -> Dynamic Trust Fusion Engine
# =========================================================

app = Flask(__name__)
app.secret_key = "sentinel_academic_prototype_key"

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

# Set SENTINEL_CLEANUP=true to delete uploads and AMMP artifacts after each analysis.
CLEANUP_AFTER_ANALYSIS = os.environ.get(
    "SENTINEL_CLEANUP", "false"
).lower() in ("1", "true", "yes")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wav', 'mp3'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

users = {}

# =========================================================
# FILE VALIDATION
# =========================================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================================================
# FFmpeg Detection
# =========================================================
def get_ffmpeg_path():

    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    win_path = r"C:\ffmpeg\bin\ffmpeg.exe"

    if os.path.exists(win_path):
        return win_path

    return None


# =========================================================
# AFCP - Adaptive Frame Consistency Profiling
# Sentinel-AMMP Video Analysis
# =========================================================
def _afcp_analyze_frames(frame_iter):

    blur_vals = []
    motion_vals = []
    static_ratios = []
    brightness_vals = []
    prev_frame = None

    for frame in frame_iter:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness_vals.append(np.mean(gray))
        blur_vals.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        if prev_frame is not None:

            diff = cv2.absdiff(prev_frame, gray)
            motion_vals.append(np.mean(diff))
            static_pixels = np.sum(diff <= 1)
            static_ratios.append(static_pixels / diff.size)

        prev_frame = gray

    return blur_vals, motion_vals, static_ratios, brightness_vals


def AFCP_video_analysis(video_path=None, frame_paths=None):

    try:

        if frame_paths:

            frames = []
            for path in sorted(frame_paths):
                frame = cv2.imread(path)
                if frame is not None:
                    frames.append(frame)

            if not frames:
                return 0.5, "ERROR", "Could not read AMMP preprocessed frames."

            blur_vals, motion_vals, static_ratios, brightness_vals = _afcp_analyze_frames(
                frames
            )

        elif video_path:

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                return 0.5, "ERROR", "Could not open video."

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_count = min(60, total_frames)
            step = max(1, total_frames // sample_count)

            def _video_frame_iter():

                frame_idx = 0
                while frame_idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
                    frame_idx += step

            blur_vals, motion_vals, static_ratios, brightness_vals = _afcp_analyze_frames(
                _video_frame_iter()
            )
            cap.release()

        else:
            return 0.5, "ERROR", "No video source provided for AFCP."

        if not blur_vals:
            return 0.5, "UNKNOWN", "Insufficient video frames."

        blur_mean = np.mean(blur_vals)

        motion_mean = np.mean(motion_vals) if motion_vals else 0

        static_mean = np.mean(static_ratios) * 100 if static_ratios else 0

        brightness_mean = np.mean(brightness_vals)

        print("\n========== AFCP VIDEO DIAGNOSTICS ==========")

        print(f"Blur Mean: {blur_mean:.2f}")

        print(f"Motion Mean: {motion_mean:.2f}")

        print(f"Static Ratio: {static_mean:.2f}%")

        print(f"Brightness Mean: {brightness_mean:.2f}")

        print("============================================")

        # =================================================
        # AFCP DECISION LOGIC
        # =================================================

        anomaly_score = 0

        # Frozen AI background
        if static_mean > 60:
            anomaly_score += 2

        # Unrealistically smooth
        if blur_mean < 30:
            anomaly_score += 2

        # Very low motion
        if motion_mean < 1.5:
            anomaly_score += 1

        # Suspicious lighting
        if brightness_mean < 40 or brightness_mean > 220:
            anomaly_score += 1

        # =================================================

        if anomaly_score >= 4:

            video_score = 0.20

            status = "SUSPICIOUS / AI"

            reason = (
                "AFCP detected severe frame consistency anomalies, "
                "frozen regions, and synthetic visual patterns."
            )

        elif anomaly_score >= 2:

            video_score = 0.45

            status = "POSSIBLY FAKE"

            reason = (
                "AFCP detected moderate visual inconsistencies "
                "associated with manipulated media."
            )

        else:

            naturalness = min(
                1.0,
                (motion_mean / 10) +
                ((100 - static_mean) / 120) +
                (blur_mean / 400)
            )

            video_score = round(
                min(0.95, 0.72 + naturalness * 0.20),
                2
            )

            status = "AUTHENTIC / REAL"

            reason = (
                "AFCP detected natural camera motion, "
                "sensor noise, and realistic frame transitions."
            )

        return float(video_score), status, reason

    except Exception as e:

        print("AFCP Video Error:", e)

        return 0.5, "ERROR", f"AFCP processing failed: {e}"


# =========================================================
# AUDIO EXTRACTION
# =========================================================
def extract_audio(video_path, audio_path):

    ffmpeg = get_ffmpeg_path()

    if not ffmpeg:
        print("FFmpeg not found.")
        return False

    try:

        command = [
            ffmpeg,
            '-i', video_path,
            '-q:a', '0',
            '-map', 'a?',
            '-y',
            audio_path
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=60
        )

        return os.path.exists(audio_path)

    except Exception as e:

        print("FFmpeg Error:", e)

        return False


# =========================================================
# SVIM - Spectral Voice Irregularity Mapping
# Delegates to svim package; scores are 0-100 (normalize for DTFE).
# =========================================================
def SVIM_audio_analysis(audio_path):
    result = analyze_audio(audio_path)
    return result.audio_score, result.status, result.reason


def _cleanup_analysis_artifacts(paths):

    for path in paths:

        if not path:
            continue

        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path) and os.path.exists(path):
            os.remove(path)


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()

        password = request.form.get("password", "")

        if username in users and users[username] == password:

            session["user"] = username

            return redirect(url_for("dashboard"))

        else:

            flash("Invalid credentials.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()

        password = request.form.get("password", "")

        if not username or not password:

            flash("Username and password are required.", "error")

        elif username in users:

            flash("Username already exists.", "error")

        else:

            users[username] = password

            flash("Registration successful. Please log in.", "success")

            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))


@app.route("/uploads/<path:storage_name>")
def serve_upload(storage_name):

    return send_from_directory(UPLOAD_FOLDER, storage_name)


def _media_type_from_filename(filename):

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in {"mp4", "avi", "mov"}:
        return "video"
    if ext in {"wav", "mp3"}:
        return "audio"
    return "video"


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        file = request.files.get("video")

        if not file or file.filename == '':

            flash("No file selected.", "error")

            return redirect(request.url)

        if not allowed_file(file.filename):

            flash("Unsupported file type.", "error")

            return redirect(request.url)

        filename = secure_filename(file.filename)

        uid = str(uuid.uuid4())[:8]

        video_path = os.path.join(
            UPLOAD_FOLDER,
            f"{uid}_{filename}"
        )

        audio_path = os.path.join(
            OUTPUT_FOLDER,
            f"{uid}.wav"
        )

        ammp_result = None
        svim_result = None
        temporal_result = None
        fusion_result = None
        cleanup_paths = [video_path, audio_path]

        try:

            file.save(video_path)

            audio_only = is_audio_only_upload(video_path)

            # =================================================
            # AMMP - Adaptive Multi-Modal Preprocessing
            # =================================================

            ammp_result = preprocess_media(
                media_path=video_path,
                output_dir=OUTPUT_FOLDER,
                uid=uid,
                extract_audio_fn=extract_audio,
            )

            if ammp_result.raw_audio_path:
                cleanup_paths.append(ammp_result.raw_audio_path)

            if ammp_result.preprocessed_audio_path:
                cleanup_paths.append(ammp_result.preprocessed_audio_path)

            if ammp_result.frames_dir:
                cleanup_paths.append(ammp_result.frames_dir)

            # =================================================
            # TEMPORAL ANALYSIS (optical flow; not fused in DTFE yet)
            # =================================================

            if audio_only:
                temporal_result = analyze_temporal()
            else:
                temporal_result = analyze_temporal(
                    video_path=video_path,
                    frame_paths=ammp_result.frame_paths or None,
                )

            # =================================================
            # AFCP VIDEO ANALYSIS
            # =================================================

            if audio_only:

                video_score = 0.5

                video_status = "NO VIDEO"

                video_reason = (
                    "Audio-only upload; video analysis skipped."
                )

                prediction = None

            else:

                prediction = predict_video(video_path)

                video_score = prediction["score"]

                video_status = prediction["status"]

                video_reason = prediction["reason"]

                video_real_probability = prediction["real_probability"]

                video_fake_probability = prediction["fake_probability"]

                video_real_votes = prediction["real_votes"]

                video_fake_votes = prediction["fake_votes"]

                video_frames = prediction["frames"]
            # =================================================
            # SVIM AUDIO ANALYSIS (uses AMMP-normalized audio when available)
            # =================================================

            svim_audio_path = ammp_result.preprocessed_audio_path

            if not svim_audio_path and not audio_only:

                if extract_audio(video_path, audio_path):
                    svim_audio_path = audio_path

            if svim_audio_path and os.path.exists(svim_audio_path):

                svim_result = analyze_audio(svim_audio_path)
                audio_score = svim_result.audio_score
                audio_status = svim_result.status
                audio_reason = svim_result.reason

            else:

                audio_score = None

                audio_status = "NO AUDIO"

                audio_reason = "No audio stream detected."

            # =================================================
            # DTFE - Trust Score Fusion (fusion package)
            # =================================================

            fusion_result = fuse_trust_scores(
                FusionInput(
                    spatial_score=None if audio_only else round(video_score, 1),
                    temporal_score=(
                        temporal_result.temporal_score
                        if temporal_result is not None
                        else None
                    ),
                    mfcc_score=svim_result.mfcc_score if svim_result else None,
                    acoustic_pattern_score=(
                        svim_result.acoustic_pattern_score if svim_result else None
                    ),
                    audio_only=audio_only,
                    has_audio=svim_result is not None,
                    has_video=not audio_only,
                )
            )

            final_score = fusion_result.fusion_final_score
            classification = FusionClassification(fusion_result.fusion_classification)
            result, status_color = dashboard_result_labels(classification)

            media_storage_name = os.path.basename(video_path)
            media_preview_url = None
            if os.path.exists(video_path):
                media_preview_url = url_for(
                    "serve_upload",
                    storage_name=media_storage_name,
                )

        finally:

            if CLEANUP_AFTER_ANALYSIS:
                _cleanup_analysis_artifacts(cleanup_paths)

        template_kwargs = dict(
            has_analysis=True,
            audio_only=audio_only,
            analysis_summary=fusion_result.summary,
            pipeline_mode=fusion_result.mode.value,
            uploaded_filename=filename,
            media_storage_name=media_storage_name,
            media_preview_url=media_preview_url,
            media_type=_media_type_from_filename(filename),
            temporal_status=(
                temporal_result.status if temporal_result is not None else "N/A"
            ),
            video_score=video_score,
            video_status=video_status,
            video_reason=video_reason,
            audio_score=audio_score,
            audio_status=audio_status,
            audio_reason=audio_reason,
            final_score=final_score,
            result=result,
            status_color=status_color,
        )

        if svim_result is not None:
            template_kwargs.update(
                svim_mfcc_score=svim_result.mfcc_score,
                svim_acoustic_pattern_score=svim_result.acoustic_pattern_score,
                svim_audio_score=svim_result.audio_score,
                svim_signal_variance=svim_result.signal_variance,
                svim_spectral_contrast=svim_result.spectral_contrast,
                svim_rms_energy=svim_result.rms_energy,
                svim_zcr=svim_result.zcr,
            )

        if temporal_result is not None:
            diag = temporal_result.diagnostics
            template_kwargs.update(
                temporal_motion_consistency_score=temporal_result.motion_consistency_score,
                temporal_stability_score=temporal_result.temporal_stability_score,
                temporal_frame_transition_score=temporal_result.frame_transition_score,
                temporal_score=temporal_result.temporal_score,
                temporal_score_normalized=temporal_result.temporal_score_normalized,
                temporal_confidence_level=temporal_result.confidence_level.value,
                temporal_mean_flow_magnitude=diag.mean_flow_magnitude,
                temporal_flow_magnitude_std=diag.flow_magnitude_std,
                temporal_mean_flow_angle=diag.mean_flow_angle,
                temporal_max_flow_magnitude=diag.max_flow_magnitude,
                temporal_flow_spike_ratio=diag.flow_spike_ratio,
                temporal_sudden_transition_count=diag.sudden_transition_count,
                temporal_frame_pair_count=diag.frame_pair_count,
                temporal_processed_frame_count=diag.processed_frame_count,
                temporal_status=temporal_result.status,
                temporal_reason=temporal_result.reason,
            )

        if fusion_result is not None:
            template_kwargs.update(
                fusion_spatial_score=fusion_result.fusion_spatial_score,
                fusion_temporal_score=fusion_result.fusion_temporal_score,
                fusion_video_score=fusion_result.fusion_video_score,
                fusion_mfcc_score=fusion_result.fusion_mfcc_score,
                fusion_acoustic_pattern_score=fusion_result.fusion_acoustic_pattern_score,
                fusion_audio_score=fusion_result.fusion_audio_score,
                fusion_final_score=fusion_result.fusion_final_score,
                fusion_classification=fusion_result.fusion_classification,
                fusion_confidence_level=fusion_result.fusion_confidence_level.value,
            )
            if prediction is not None:
                template_kwargs.update(
                    video_real_probability=prediction["real_probability"],
                    video_fake_probability=prediction["fake_probability"],
                    video_real_votes=prediction["real_votes"],
                    video_fake_votes=prediction["fake_votes"],
                    video_frames=prediction["frames"],
                )
        return render_template("dashboard.html", **template_kwargs)

    return render_template(
        "dashboard.html",
        has_analysis=False,
    )


if __name__ == "__main__":
    app.run(debug=True)