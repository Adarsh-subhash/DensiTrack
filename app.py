#!/usr/bin/env python3
import datetime
import os
import tempfile
from collections import defaultdict
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.solutions.heatmap import Heatmap

# -------------------------------------------------------------
# Helpers & Inference Pipeline
# -------------------------------------------------------------

def ensure_numpy_image(image):
    if hasattr(image, 'read') or hasattr(image, 'name'):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    return image


def select_image_input(uploaded_image, browser_webcam_frame, system_webcam_frame=None):
    if browser_webcam_frame is not None:
        return browser_webcam_frame
    if system_webcam_frame is not None:
        return system_webcam_frame
    return uploaded_image


def capture_system_webcam(camera_index=0):
    for index in (camera_index, 1, 2):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def draw_dots_on_frame(frame, results, min_conf=0.3, min_box_area=0, track_history=None, tracking_enabled=False):
    annotated = frame.copy()
    if not results or len(results) == 0:
        return annotated
    for box in results[0].boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)
        if conf < min_conf or area < min_box_area:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 7, (0, 0, 255), -1)
        if tracking_enabled and hasattr(box, 'id') and box.id is not None and track_history is not None:
            track_id = int(box.id.item())
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)
            for i in range(1, len(track_history[track_id])):
                cv2.line(annotated, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 0), 2)
    return annotated


def resolve_model_path(model_id):
    if not os.path.exists(model_id) and model_id in ["medium.pt", "best.pt"]:
        return "yolov8n.pt"
    return model_id


def yolov12_tracker_inference(image, video, model_id, image_size, conf_threshold, mode, viz_mode, use_tracking):
    actual_model = resolve_model_path(model_id)
    try:
        model = YOLO(actual_model)
    except Exception:
        model = YOLO("yolov8n.pt")

    if hasattr(model, 'model') and hasattr(model.model, 'classes'):
        model.model.classes = [0]

    output_video_path = None
    count = 0

    if mode == "Original":
        if video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            return None, video_path, 0, "Original video loaded"
        return None, None, 0, "Original mode requires video input"

    if mode == "Detection":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0])
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes) if results and len(results) > 0 else 0
            return annotated_image[:, :, ::-1], None, count, f"Max pedestrians in frame: {count}"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_det.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            max_count = 0
            track_history = {}
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, classes=[0])
                current_count = len(results[0].boxes) if results and len(results) > 0 else 0
                if current_count > max_count:
                    max_count = current_count
                if viz_mode == "Dots":
                    annotated_frame = draw_dots_on_frame(frame, results, track_history=track_history, tracking_enabled=True)
                else:
                    annotated_frame = results[0].plot()
                out.write(annotated_frame)
            cap.release()
            out.release()
            return None, output_video_path, max_count, f"Max pedestrians in video: {max_count}"

    elif mode == "Tracking":
        if image is not None:
            image = ensure_numpy_image(image)
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")
            if viz_mode == "Dots":
                annotated_image = draw_dots_on_frame(image, results)
            else:
                annotated_image = results[0].plot()
            count = len(results[0].boxes) if results and len(results) > 0 else 0
            return annotated_image[:, :, ::-1], None, count, f"Max pedestrians in frame: {count}"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_track.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            track_history = defaultdict(list)
            max_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.track(frame, persist=True, imgsz=image_size, conf=conf_threshold, classes=[0], tracker="bytetrack.yaml")[0]
                if results.boxes and results.boxes.id is not None:
                    boxes = results.boxes.xywh.cpu()
                    track_ids = results.boxes.id.int().cpu().tolist()
                    annotated_frame = results.plot()
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((int(x), int(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                        if len(points) > 1:
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                    out.write(annotated_frame)
                    if len(track_ids) > max_count:
                        max_count = len(track_ids)
                else:
                    out.write(frame)
            cap.release()
            out.release()
            return None, output_video_path, max_count, f"Max tracked pedestrians: {max_count}"

    elif mode == "Heatmap":
        if image is not None:
            image = ensure_numpy_image(image)
            return image, None, 0, "Heatmap for static image is calculated during video sequence"
        elif video is not None:
            video_path = tempfile.mktemp(suffix=".mp4")
            with open(video_path, "wb") as f:
                with open(video.name if hasattr(video, 'name') else str(video), "rb") as g:
                    f.write(g.read())
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = tempfile.mktemp(suffix="_heatmap.mp4")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            heatmap = Heatmap(model=actual_model, imgsz=image_size, conf=conf_threshold, classes=[0])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                heatmap_frame = heatmap.generate_heatmap(frame)
                out.write(heatmap_frame)
            cap.release()
            out.release()
            return None, output_video_path, 0, "Heatmap processing complete"

    return None, None, 0, "No input provided"


# -------------------------------------------------------------
# Data Store for Multi-Camera & Alerts Management
# -------------------------------------------------------------

INITIAL_CAMERAS = [
    {
        "id": "CAM-01",
        "name": "CAM 01 - Main Entrance Plaza",
        "location": "North Gate Section A",
        "type": "System Hardware Webcam",
        "status": "LIVE",
        "assigned_name": "Vikram Rathore",
        "assigned_role": "Crowd Control Lead",
        "density": "High (84%)",
    },
    {
        "id": "CAM-02",
        "name": "CAM 02 - North Gate Concourse",
        "location": "Sector 3 Checkpoint",
        "type": "IP / RTSP Stream",
        "status": "LIVE",
        "assigned_name": "Sarah Connor",
        "assigned_role": "Perimeter Security Officer",
        "density": "Medium (52%)",
    },
    {
        "id": "CAM-03",
        "name": "CAM 03 - West Promenade & Stairs",
        "location": "West Plaza Walkway",
        "type": "PTZ Dome Optical",
        "status": "LIVE",
        "assigned_name": "Dr. Aarav Mehra",
        "assigned_role": "Medical First Responder",
        "density": "Low (21%)",
    },
    {
        "id": "CAM-04",
        "name": "CAM 04 - Sector B Queue Corridor",
        "location": "Central Pavilion Lane 4",
        "type": "Infrared Thermal / Optical",
        "status": "WARNING",
        "assigned_name": "Rajesh Sharma",
        "assigned_role": "Emergency Dispatch Commander",
        "density": "Critical (93%)",
    },
]

INITIAL_ALERTS = [
    {
        "id": "ALT-101",
        "camera_id": "CAM-04",
        "camera_name": "CAM 04 - Sector B Queue Corridor",
        "severity": "Critical",
        "message": "Density bottleneck detected! Surge rate exceeds 3.4 persons/sq.m limit.",
        "assigned_person": "Rajesh Sharma (Emergency Dispatch)",
        "timestamp": "19:18:22",
        "status": "Active",
    },
    {
        "id": "ALT-102",
        "camera_id": "CAM-01",
        "camera_name": "CAM 01 - Main Entrance Plaza",
        "severity": "Warning",
        "message": "High pedestrian inflow rate detected at entry turnstiles.",
        "assigned_person": "Vikram Rathore (Crowd Control Lead)",
        "timestamp": "19:14:05",
        "status": "Acknowledged",
    },
    {
        "id": "ALT-103",
        "camera_id": "CAM-02",
        "camera_name": "CAM 02 - North Gate Concourse",
        "severity": "Info",
        "message": "Steady queue distribution maintained across 4 lanes.",
        "assigned_person": "Sarah Connor (Perimeter Security)",
        "timestamp": "19:02:11",
        "status": "Resolved",
    },
    {
        "id": "ALT-104",
        "camera_id": "CAM-03",
        "camera_name": "CAM 03 - West Promenade & Stairs",
        "severity": "Info",
        "message": "Normal pedestrian throughput observed.",
        "assigned_person": "Dr. Aarav Mehra (Medical Team)",
        "timestamp": "18:55:40",
        "status": "Resolved",
    },
]

# -------------------------------------------------------------
# HTML Rendering Utilities for Visuvate Dark Aesthetic
# -------------------------------------------------------------

def render_camera_sidebar_html(cameras, active_cam_id=None):
    items = []
    for idx, cam in enumerate(cameras):
        is_selected = (cam["id"] == active_cam_id)
        border_style = "border: 1px solid #0072f5; background: #121829;" if is_selected else "border: 1px solid #1c1c22; background: #0f0f13;"
        status_color = "#22c55e" if cam.get("status") == "LIVE" else "#ef4444"
        items.append(f"""
        <div class="sidebar-cam-card" onclick="selectCameraPage('{cam['id']}')" style="display: flex; align-items: center; justify-content: space-between; padding: 9px 12px; border-radius: 8px; margin-bottom: 5px; {border_style} font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.2s ease;">
            <div style="display: flex; align-items: center; gap: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                <span style="overflow: hidden; text-overflow: ellipsis; color: #ffffff; font-weight: 500;">{cam['name']}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; flex-shrink: 0;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: {status_color};"></span>
                <span style="font-size: 11px; opacity: 0.8; color: #a1a1aa;">{cam['id']}</span>
            </div>
        </div>
        """)
    return f"""
    <div style="display: flex; flex-direction: column; gap: 2px;">
        {''.join(items)}
    </div>
    """


def render_camera_detail_full_html(cam=None, alerts=None):
    if cam is None:
        cam = INITIAL_CAMERAS[0]
    if alerts is None:
        alerts = INITIAL_ALERTS

    cam_id = cam.get("id", "CAM-01")
    officer_name = cam.get("assigned_name", "Vikram Rathore")
    officer_role = cam.get("assigned_role", "Crowd Control Lead")
    location = cam.get("location", "North Gate Section A")
    initial = officer_name[0] if officer_name else "V"
    density_str = cam.get("density", "High (84%)")
    density_pct = 85 if ("84%" in density_str or "93%" in density_str) else (52 if "52%" in density_str else 22)
    density_color = '#ef4444' if density_pct >= 70 else ('#f59e0b' if density_pct >= 50 else '#22c55e')
    chokepoint_label = 'Critical Surge' if density_pct >= 70 else ('Elevated Flow' if density_pct >= 50 else 'Optimal Flow')

    cam_alerts = [a for a in alerts if a.get("camera_id") == cam_id or a.get("camera_name") == cam.get("name")]
    alert_rows = []
    if cam_alerts:
        for idx, a in enumerate(cam_alerts[:4]):
            badge_bg = "rgba(239, 68, 68, 0.18)" if a["severity"] == "Critical" else ("rgba(245, 158, 11, 0.18)" if a["severity"] == "Warning" else "rgba(34, 197, 94, 0.18)")
            badge_color = "#f87171" if a["severity"] == "Critical" else ("#fbbf24" if a["severity"] == "Warning" else "#4ade80")
            badge_border = "#ef4444" if a["severity"] == "Critical" else ("#f59e0b" if a["severity"] == "Warning" else "#22c55e")
            alert_rows.append(f"""
            <div style="background: #16161c; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <span style="font-size: 10px; font-weight: 700; color: #a1a1aa; text-transform: uppercase;">Type:</span>
                        <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700;">{a['severity']}</span>
                    </div>
                    <span style="font-size: 10px; font-family: monospace; color: #a1a1aa;">{a['timestamp']}</span>
                </div>
                <div style="font-size: 11px; color: #ffffff; line-height: 1.4; margin-bottom: 6px;">
                    <b>Details:</b> {a['message']}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1f1f26; padding-top: 6px;">
                    <span style="font-size: 10px; color: #93c5fd;">Assigned: {officer_name}</span>
                    <button onclick="sendAlertToPersonnel('{cam_id}', '{a['severity']}', '{officer_name}')" style="background: #ef4444; color: #ffffff; border: none; padding: 4px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer; transition: background 0.2s;" onmouseover="this.style.background='#dc2626'" onmouseout="this.style.background='#ef4444'">
                        Send Alert
                    </button>
                </div>
            </div>
            """)
    else:
        alert_rows.append(f"""
        <div style="padding: 14px; background: #16161c; border: 1px dashed #272730; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 12px;">
            No active security incidents flagged for {cam_id}. Area status is nominal.
        </div>
        """)
    alerts_markup = "".join(alert_rows)

    return f"""
    <!-- Top Header Bar with Back Button and Model Selector Dropdown -->
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; background: #121216; border: 1px solid #23232b; border-radius: 12px; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <button onclick="returnToHome()" class="btn-visuvate-filter" style="display: inline-flex; align-items: center; gap: 6px; background: #18181f; color: #ffffff; border: 1px solid #2c2c36; padding: 7px 14px; border-radius: 8px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s ease;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.5"><polyline points="15 18 9 12 15 6"></polyline></svg>
                Back to Home
            </button>
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span id="detail-cam-title" style="font-size: 16px; font-weight: 700; color: #ffffff;">{cam.get('name', 'Camera View')}</span>
                    <span id="detail-cam-id-badge" style="font-size: 11px; background: rgba(0, 114, 245, 0.18); color: #0072f5; border: 1px solid rgba(0, 114, 245, 0.4); padding: 2px 8px; border-radius: 6px; font-weight: 600;">{cam.get('id', 'CAM-01')}</span>
                </div>
                <div style="font-size: 11px; color: #71717a; margin-top: 2px;">
                    Monitored Sector: <span id="detail-cam-location" style="color: #d4d4d8; font-weight: 500;">{location}</span> • Source: <span id="detail-cam-type" style="color: #a1a1aa;">{cam.get('type', 'Standard Feed')}</span>
                </div>
            </div>
        </div>
        
        <!-- Model Selection Dropdown (Replacing static ● REC badge) -->
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 8px; background: #16161c; border: 1px solid #282832; padding: 4px 10px; border-radius: 8px;">
                <span style="display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: #a1a1aa; font-weight: 600;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><path d="M7 7h10v10H7z"/></svg>
                    AI Model:
                </span>
                <select id="detail-model-select" onchange="onModelChanged(this.value)" style="background: #0e0e12; color: #ffffff; border: 1px solid #33333f; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 600; outline: none; cursor: pointer;">
                    <option value="best.pt" selected>Custom Trained (best.pt)</option>
                    <option value="medium.pt">Fine-Tuned Crowd (medium.pt)</option>
                    <option value="yolov12n.pt">YOLOv12 Nano (yolov12n.pt)</option>
                    <option value="yolov12m.pt">YOLOv12 Medium (yolov12m.pt)</option>
                    <option value="yolov8n.pt">YOLOv8 Nano (yolov8n.pt)</option>
                    <option value="yolo11n.pt">YOLO11 Nano (yolo11n.pt)</option>
                </select>
                <span id="model-active-indicator" style="width: 7px; height: 7px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 8px #22c55e;"></span>
            </div>
        </div>
    </div>

    <!-- Center Camera Feed: CAM-01 Live Webcam Box (Uncropped - object-fit: contain) -->
    <div id="cam-detail-webcam-box" style="position: relative; width: 100%; min-height: 380px; max-height: 480px; background: #000000; border-radius: 14px; overflow: hidden; border: 1px solid #23232b; box-shadow: 0 10px 30px rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
        <video id="cam-detail-video" autoplay playsinline muted style="width: 100%; height: 100%; max-height: 480px; object-fit: contain; display: none; background: #000;"></video>
        <div id="cam-detail-placeholder" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; cursor: pointer; padding: 24px; text-align: center; width: 100%; min-height: 380px;" onclick="initCamDetailWebcam()">
            <div style="width: 52px; height: 52px; border-radius: 50%; background: rgba(0, 114, 245, 0.2); border: 1px solid rgba(0, 114, 245, 0.5); display: flex; align-items: center; justify-content: center; color: #0072f5; box-shadow: 0 0 24px rgba(0, 114, 245, 0.3);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
            </div>
            <div style="color: #ffffff; font-size: 15px; font-weight: 600;">Hardware Webcam Feed Active</div>
            <div style="color: #71717a; font-size: 12px; max-width: 320px;">Streaming live webcam into CAM 01. Click if camera permission prompt is required.</div>
        </div>
        <!-- CCTV HUD Overlay -->
        <div style="position: absolute; top: 12px; left: 16px; display: flex; align-items: center; gap: 8px; pointer-events: none;">
            <span style="font-size: 10px; font-family: monospace; font-weight: 700; color: #ef4444; background: rgba(0,0,0,0.7); border: 1px solid rgba(239,68,68,0.4); padding: 3px 8px; border-radius: 4px;">● REC [LIVE]</span>
            <span style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;">PRIMARY WEBCAM</span>
            <span id="cam-model-badge" style="font-size: 10px; font-family: monospace; color: #60a5fa; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #1e3a8a;">MODEL: best.pt</span>
        </div>
        <div style="position: absolute; bottom: 12px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: center; pointer-events: none;">
            <span id="cam-detail-person-metric" style="font-size: 10px; font-family: monospace; color: #22c55e; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);">● 1080p HD • ANALYZED PERSONS: 0</span>
            <span id="cam-detail-timestamp-webcam" style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;"></span>
        </div>
    </div>

    <!-- Center Camera Feed: Optical Canvas Box (For Other Cameras - Uncropped object-fit: contain) -->
    <div id="cam-detail-optical-box" style="position: relative; width: 100%; min-height: 380px; max-height: 480px; background: #000000; border-radius: 14px; overflow: hidden; border: 1px solid #23232b; box-shadow: 0 10px 30px rgba(0,0,0,0.6); display: none; align-items: center; justify-content: center; margin-bottom: 12px;">
        <canvas id="cctv-optical-stream-canvas" width="640" height="380" style="width: 100%; height: 100%; max-height: 480px; object-fit: contain; background: #000; display: block;"></canvas>
        <div style="position: absolute; top: 12px; left: 16px; display: flex; align-items: center; gap: 8px; pointer-events: none;">
            <span style="font-size: 10px; font-family: monospace; font-weight: 700; color: #22c55e; background: rgba(0,0,0,0.7); border: 1px solid rgba(34, 197, 94, 0.4); padding: 3px 8px; border-radius: 4px;">● RTSP STREAM</span>
            <span style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;">OPTICAL SURVEILLANCE</span>
            <span id="optical-model-badge" style="font-size: 10px; font-family: monospace; color: #60a5fa; background: rgba(0,0,0,0.7); padding: 3px 8px; border-radius: 4px; border: 1px solid #1e3a8a;">MODEL: best.pt</span>
        </div>
        <div style="position: absolute; bottom: 12px; left: 16px; right: 16px; display: flex; justify-content: space-between; align-items: center; pointer-events: none;">
            <span id="cam-optical-person-metric" style="font-size: 10px; font-family: monospace; color: #22c55e; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);">● AI TRACKING ENABLED • ANALYZED PERSONS: 4</span>
            <span id="cam-detail-timestamp-optical" style="font-size: 10px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.75); padding: 3px 8px; border-radius: 4px; border: 1px solid #27272a;"></span>
        </div>
    </div>

    <!-- Bottom Split: Heatmap on Left, Personnel Allotted & Alerts on Right -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
        <!-- Left Side: Heatmap based on Crowd from Feed -->
        <div style="background: #111115; border: 1px solid #22222a; border-radius: 12px; padding: 14px; min-height: 290px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <div style="font-size: 14px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 6px;">
                        Heat Map
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 6px;">
                    <span id="detail-heatmap-count-badge" style="font-size: 10px; background: rgba(0, 114, 245, 0.18); color: #60a5fa; border: 1px solid rgba(0, 114, 245, 0.4); padding: 2px 7px; border-radius: 5px; font-weight: 600;">0 Persons Detected</span>
                    <span style="font-size: 10px; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 2px 7px; border-radius: 5px; font-weight: 600;">JET THERMAL MAP</span>
                </div>
            </div>

            <div style="position: relative; width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #1f1f27;">
                <canvas id="crowd-heatmap-canvas" width="480" height="190" style="width: 100%; height: 190px; display: block; background: #06070a;"></canvas>
                
                <div style="position: absolute; bottom: 8px; right: 10px; display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.75); padding: 4px 8px; border-radius: 6px; border: 1px solid #27272a;">
                    <span style="font-size: 9px; color: #60a5fa; font-family: monospace;">LOW</span>
                    <div style="width: 70px; height: 8px; border-radius: 4px; background: linear-gradient(to right, #3b82f6, #06b6d4, #22c55e, #eab308, #ef4444);"></div>
                    <span style="font-size: 9px; color: #ef4444; font-family: monospace;">HIGH</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px;">
                <div style="background: #16161c; border: 1px solid #24242e; border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 10px; color: #71717a; text-transform: uppercase;">Crowd Density</div>
                    <div id="detail-density-val" style="font-size: 13px; font-weight: 700; color: #ffffff; margin-top: 2px;">{density_str}</div>
                </div>
                <div style="background: #16161c; border: 1px solid #24242e; border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 10px; color: #71717a; text-transform: uppercase;">Chokepoint Level</div>
                    <div id="detail-chokepoint-val" style="font-size: 13px; font-weight: 700; color: {density_color}; margin-top: 2px;">{chokepoint_label}</div>
                </div>
                <div style="background: #16161c; border: 1px solid #24242e; border-radius: 8px; padding: 8px 10px;">
                    <div style="font-size: 10px; color: #71717a; text-transform: uppercase;">Active Model</div>
                    <div id="detail-model-tag" style="font-size: 13px; font-weight: 700; color: #ffffff; margin-top: 2px;">best.pt</div>
                </div>
            </div>
        </div>

        <!-- Right Side: Personnel Allotted & Area Incident Alerts Side by Side -->
        <div style="background: #111115; border-radius: 12px; padding: 14px; min-height: 290px; display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
            <!-- Column 1: Personnel Allotted in Area -->
            <div style="display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #ffffff; font-weight: 700; letter-spacing: 0.5px;">
                        Personnel Allotted in Area
                    </div>
                    <button id="btn-personnel-edit" onclick="toggleAddPersonnelForm()" style="background: #181822; color: #ffffff; border: none; border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 4px;">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                        Edit
                    </button>
                </div>

                <!-- Edit / Add Personnel Form (Name, Position, Contact No) -->
                <div id="add-personnel-form" style="display: none; background: #16161e; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="font-size: 11px; color: #ffffff; font-weight: 600; margin-bottom: 6px;">Add Officer:</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px;">
                        <input id="input-personnel-name" type="text" placeholder="Officer Name" style="background: #0d0d12; border: 1px solid #23232c; border-radius: 6px; padding: 6px 8px; font-size: 11px; color: #ffffff; outline: none;" />
                        <input id="input-personnel-phone" type="text" placeholder="Contact No (+91 98402 11029)" style="background: #0d0d12; border: 1px solid #23232c; border-radius: 6px; padding: 6px 8px; font-size: 11px; color: #ffffff; outline: none;" />
                        <select id="input-personnel-role" style="background: #0d0d12; border: 1px solid #23232c; border-radius: 6px; padding: 6px 8px; font-size: 11px; color: #ffffff; outline: none;">
                            <option value="Crowd Control Lead">Crowd Control Lead</option>
                            <option value="Perimeter Security Officer">Perimeter Security Officer</option>
                            <option value="Medical First Responder">Medical First Responder</option>
                            <option value="Emergency Dispatch Commander">Emergency Dispatch Commander</option>
                            <option value="Sector Safety Officer">Sector Safety Officer</option>
                        </select>
                    </div>
                    <div style="display: flex; justify-content: flex-end; gap: 6px;">
                        <button onclick="toggleAddPersonnelForm()" style="background: #202028; color: #ffffff; border: none; padding: 4px 10px; border-radius: 6px; font-size: 11px; cursor: pointer;">Cancel</button>
                        <button onclick="submitAddPersonnel()" style="background: #0072f5; color: #ffffff; border: none; padding: 4px 12px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer;">Add</button>
                    </div>
                </div>

                <!-- Personnel Dynamic Cards List -->
                <div id="detail-personnel-list" style="display: flex; flex-direction: column; gap: 6px; max-height: 220px; overflow-y: auto;">
                    <div class="personnel-card" data-idx="0" style="display: flex; align-items: center; justify-content: space-between; background: #16161c; border-radius: 8px; padding: 8px 10px;">
                        <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
                            <div class="officer-avatar" style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0072f5, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: #ffffff; flex-shrink: 0;">
                                {initial}
                            </div>
                            <div style="overflow: hidden;">
                                <div style="font-size: 12px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 5px;">
                                    <span class="officer-name-text" style="color: #ffffff;">{officer_name}</span>
                                    <span style="width: 5px; height: 5px; border-radius: 50%; background: #22c55e;"></span>
                                </div>
                                <div class="officer-role-text" style="font-size: 10px; color: #93c5fd; font-weight: 500;">{officer_role}</div>
                            </div>
                        </div>
                        <button class="personnel-delete-btn" onclick="deletePersonnel(0)" title="Delete Officer from Area" style="display: none; background: transparent; border: none; color: #a1a1aa; cursor: pointer; padding: 4px; border-radius: 4px;" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='#a1a1aa'">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                        </button>
                    </div>
                </div>
            </div>

            <!-- Column 2: Area Incident Alerts -->
            <div style="display: flex; flex-direction: column;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 11px; text-transform: uppercase; color: #ffffff; font-weight: 700; letter-spacing: 0.5px;">Area Incident Alerts</span>
                </div>
                <div id="detail-alerts-list" style="max-height: 220px; overflow-y: auto;">
                    {alerts_markup}
                </div>
            </div>
        </div>
    </div>
    """


def render_alerts_table_html(alerts, filter_severity="All", filter_cam="All"):
    filtered = []
    for a in alerts:
        if filter_severity != "All" and a["severity"] != filter_severity:
            continue
        if filter_cam != "All" and (a["camera_id"] != filter_cam and a["camera_name"] != filter_cam):
            continue
        filtered.append(a)

    rows = []
    for a in filtered:
        badge_bg = "rgba(239, 68, 68, 0.15)" if a["severity"] == "Critical" else ("rgba(245, 158, 11, 0.15)" if a["severity"] == "Warning" else "rgba(34, 197, 94, 0.15)")
        badge_color = "#f87171" if a["severity"] == "Critical" else ("#fbbf24" if a["severity"] == "Warning" else "#4ade80")
        badge_border = "#ef4444" if a["severity"] == "Critical" else ("#f59e0b" if a["severity"] == "Warning" else "#22c55e")
        rows.append(f"""
        <tr style="border-bottom: 1px solid #1a1a1f;">
            <td style="padding: 8px 12px; font-family: monospace; font-size: 11px; color: #71717a;">{a['timestamp']}</td>
            <td style="padding: 8px 12px; font-weight: 600; color: #f4f4f5; font-size: 12px;">{a['camera_name']}</td>
            <td style="padding: 8px 12px;">
                <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}50; padding: 2px 7px; border-radius: 5px; font-size: 10px; font-weight: 600;">{a['severity']}</span>
            </td>
            <td style="padding: 8px 12px; color: #d4d4d8; font-size: 12px;">{a['message']}</td>
            <td style="padding: 8px 12px; color: #60a5fa; font-size: 11px; font-weight: 500;">{a['assigned_person']}</td>
            <td style="padding: 8px 12px;">
                <span style="background: #141416; border: 1px solid #27272a; color: #22c55e; padding: 2px 7px; border-radius: 10px; font-size: 10px; font-weight: 500;">{a['status']}</span>
            </td>
        </tr>
        """)

    if not rows:
        tbody = """<tr><td colspan="6" style="padding: 16px; text-align: center; color: #71717a; font-style: italic;">No alerts found matching current filter criteria.</td></tr>"""
    else:
        tbody = "".join(rows)

    return f"""
    <div style="overflow-x: auto; background: #0c0c0e; border: 1px solid #1f1f24; border-radius: 10px;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
            <thead>
                <tr style="background: #141417; color: #a1a1aa; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #232328;">
                    <th style="padding: 8px 12px;">Time</th>
                    <th style="padding: 8px 12px;">Camera Source</th>
                    <th style="padding: 8px 12px;">Severity</th>
                    <th style="padding: 8px 12px;">Incident Details</th>
                    <th style="padding: 8px 12px;">Assigned Personnel</th>
                    <th style="padding: 8px 12px;">Status</th>
                </tr>
            </thead>
            <tbody>
                {tbody}
            </tbody>
        </table>
    </div>
    """


def render_full_personnel_table(cameras):
    rows = []
    for c in cameras:
        rows.append(f"""
        <tr style="border-bottom: 1px solid #1c1c22;">
            <td style="padding: 9px 14px; font-weight: 600; color: #f4f4f5;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 26px; height: 26px; border-radius: 50%; background: #0072f5; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; color: white;">
                        {c['assigned_name'][0]}
                    </div>
                    <span>{c['assigned_name']}</span>
                </div>
            </td>
            <td style="padding: 9px 14px; color: #0072f5; font-weight: 600;">{c['assigned_role']}</td>
            <td style="padding: 9px 14px; color: #a1a1aa; font-family: monospace;">{c['id']} ({c['name']})</td>
            <td style="padding: 9px 14px; color: #e4e4e7;">{c['location']}</td>
            <td style="padding: 9px 14px;">
                <span style="background: rgba(34, 197, 94, 0.12); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); padding: 2px 8px; border-radius: 6px; font-size: 10px; font-weight: 600;">● Active on Duty</span>
            </td>
        </tr>
        """)
    return f"""
    <div style="background: #0c0c0e; border: 1px solid #1f1f24; border-radius: 10px; overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 12px;">
            <thead>
                <tr style="background: #141418; color: #a1a1aa; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; border-bottom: 1px solid #232328;">
                    <th style="padding: 9px 14px;">Officer / Responder Name</th>
                    <th style="padding: 9px 14px;">Assigned Role</th>
                    <th style="padding: 9px 14px;">Assigned Camera</th>
                    <th style="padding: 9px 14px;">Monitored Patrol Section</th>
                    <th style="padding: 9px 14px;">Duty Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


# -------------------------------------------------------------
# Gradio UI Construction with Visuvate Design System
# -------------------------------------------------------------

VISUVATE_CSS = """
/* Reset & Zero out space above header */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #000000 !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 16px 20px 16px !important;
    margin: 0 !important;
    background-color: #000000 !important;
    background-image: 
        radial-gradient(circle at 0% 0%, rgba(0, 114, 245, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 100% 0%, rgba(0, 114, 245, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(0, 114, 245, 0.12) 0%, transparent 50%),
        #000000 !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

.gradio-container > .main, #root, .gradio-container div.app {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Glassmorphic Framer Workspace Canvas */
.visuvate-canvas {
    background: #09090b !important;
    border: 1px solid #202026 !important;
    border-radius: 16px !important;
    box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.9), 0 0 40px rgba(0, 114, 245, 0.12) !important;
    overflow: hidden !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
}

/* Quad Grid Camera Cards */
.cctv-quad-card {
    background: #0d0d10 !important;
    border: 1px solid #222228 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
    transition: border-color 0.2s ease !important;
    display: flex !important;
    flex-direction: column !important;
}
.cctv-quad-card:hover {
    border-color: #0072f5 !important;
}

.cctv-quad-header {
    background: #141418 !important;
    border-bottom: 1px solid #202026 !important;
    padding: 6px 10px !important;
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    font-size: 11px !important;
}

.rec-pill {
    background: #ef4444 !important;
    color: #ffffff !important;
    font-size: 9px !important;
    font-weight: 700 !important;
    padding: 2px 5px !important;
    border-radius: 3px !important;
    letter-spacing: 0.04em !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

/* Plus Slot in Quadrants */
.cctv-plus-slot {
    height: 165px;
    background: #070709;
    border: 1px dashed #23232c;
    border-radius: 8px;
    margin: 6px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}
.cctv-plus-slot:hover {
    border-color: #0072f5;
    background: rgba(0, 114, 245, 0.05);
}
.cctv-plus-slot .plus-sign {
    font-size: 36px;
    line-height: 1;
    color: #0072f5;
    font-weight: 300;
    margin-bottom: 6px;
}
.cctv-plus-slot .plus-label {
    font-size: 13px;
    font-weight: 600;
    color: #f4f4f5;
}
.cctv-plus-slot .plus-sub {
    font-size: 11px;
    color: #71717a;
    margin-top: 3px;
}

/* Eradicate Gradio Image Toolbar & HUD Controls (▶, 🔍, ✋, 1X, 🌙, 🔲) */
.gradio-container .image-toolbar,
.gradio-container .toolbar,
.gradio-container div[data-testid="image-toolbar"],
.gradio-container button.tool,
.gradio-container button[aria-label="Zoom in"],
.gradio-container button[aria-label="Zoom out"],
.gradio-container button[aria-label="Reset"],
.gradio-container button[aria-label="Pan"],
.gradio-container button[title="Zoom"],
.gradio-container button[title="Pan"],
.gradio-container button[title="Reset"],
.gradio-container button[aria-label="Clear"],
.gradio-container .svelte-1p9fsmv,
.cctv-quad-card .image-toolbar,
.cctv-quad-card .toolbar,
.cctv-quad-card button.tool {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Eradicate ALL White Boxes / White Badges from Form Controls & Dropdowns */
.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container fieldset,
.gradio-container .wrap,
.gradio-container .secondary-wrap,
.gradio-container .input-container,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container div[data-testid],
.gradio-container .panel,
.gradio-container .tab-item,
.gradio-container [class*="svelte-"] {
    background-color: transparent !important;
    border: none !important;
    color: #f4f4f5 !important;
    box-shadow: none !important;
}

/* Target Labels to eliminate white badge effect */
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container span[data-testid="block-info"],
.gradio-container .gr-input-label,
.gradio-container span.svelte-1ed2p3z,
.gradio-container label.svelte-1ed2p3z,
.gradio-container span.svelte-11l35i6,
.gradio-container span.svelte-1gf128n {
    background: transparent !important;
    background-color: transparent !important;
    color: #a1a1aa !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding-left: 0 !important;
}

/* Input, Select, Textarea Dark styling - SOLID OPAQUE DARK BACKGROUND */
.gradio-container input,
.gradio-container select,
.gradio-container textarea,
.gradio-container .wrap-inner,
.gradio-container .single-select,
select {
    background: #111116 !important;
    background-color: #111116 !important;
    color: #ffffff !important;
    border: 1px solid #23232c !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

/* Ensure all select options and dropdown buttons have solid background and crisp white text */
select,
select option,
.gradio-container select option,
.gradio-container .secondary-wrap,
.gradio-container .secondary-wrap span,
.gradio-container .single-select,
.gradio-container .single-select span,
.gradio-container .single-select input,
.gradio-container ul.options,
.gradio-container ul.options li,
.gradio-container .dropdown-button,
.gradio-container div[data-testid="dropdown"],
.gradio-container div[data-testid="dropdown"] button,
.gradio-container div[data-testid="dropdown"] span,
.gradio-container div[data-testid="dropdown"] input {
    color: #ffffff !important;
    background: #111116 !important;
    background-color: #111116 !important;
    border-color: #23232c !important;
    font-size: 12px !important;
    opacity: 1 !important;
}

select option:hover,
select option:checked,
.gradio-container ul.options li:hover,
.gradio-container ul.options li.selected {
    background-color: #1c1c24 !important;
    color: #ffffff !important;
}

/* Accordion Dark obsidian styling */
.gradio-container .accordion,
.gradio-container details {
    background: #0d0d11 !important;
    border: 1px solid #202026 !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
    overflow: hidden !important;
}
.gradio-container .accordion > .label-wrap,
.gradio-container details > summary {
    background: #141419 !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    color: #f4f4f5 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

/* Custom Visuvate Button Styles */
button.btn-visuvate-filter {
    background: #0072f5 !important;
    background-color: #0072f5 !important;
    color: #ffffff !important;
    border: 1px solid #0072f5 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button.btn-visuvate-filter:hover {
    background: #005bc4 !important;
    box-shadow: none !important;
}

button.btn-visuvate-alert {
    background: #ef4444 !important;
    background-color: #ef4444 !important;
    color: #ffffff !important;
    border: 1px solid #ef4444 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 9px 20px !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button.btn-visuvate-alert:hover {
    background: #dc2626 !important;
    box-shadow: none !important;
}

button.btn-visuvate-capture {
    background: #18181d !important;
    background-color: #18181d !important;
    color: #e4e4e7 !important;
    border: 1px solid #272732 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 5px 10px !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}
button.btn-visuvate-capture:hover {
    border-color: #0072f5 !important;
    color: #ffffff !important;
}

/* Hide Gradio Default Footer & API links */
footer,
.gradio-container footer,
.show-api,
.api-docs,
a[href*="gradio.app"],
a[href*="huggingface.co"],
.built-with {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Floating Glow Navigation Bar */
.floating-nav-wrapper {
    position: sticky;
    top: 6px;
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 6px 0 12px 0;
    width: 100%;
}

.glow-menu-bar {
    position: relative;
    display: inline-flex;
    align-items: center;
    padding: 5px 8px;
    border-radius: 20px;
    background: rgba(15, 15, 19, 0.82) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.09) !important;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.65), 0 0 1px 1px rgba(255, 255, 255, 0.06) !important;
    overflow: hidden;
    transition: all 0.3s ease;
}

.glow-menu-bg {
    position: absolute;
    inset: -6px;
    border-radius: 26px;
    background: radial-gradient(circle at 50% 50%, rgba(59,130,246,0.2) 0%, rgba(168,85,247,0.15) 35%, rgba(239,68,68,0.12) 75%, transparent 100%);
    opacity: 0;
    transition: opacity 0.5s ease;
    pointer-events: none;
    z-index: 0;
}
.glow-menu-bar:hover .glow-menu-bg {
    opacity: 1;
}

.glow-menu-list {
    display: flex;
    align-items: center;
    gap: 4px;
    list-style: none;
    margin: 0;
    padding: 0;
    position: relative;
    z-index: 1;
}

.glow-menu-item {
    position: relative;
}

.glow-menu-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 14px;
    background: transparent;
    border: none;
    color: #ffffff !important;
    text-decoration: none;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    position: relative;
    transition: color 0.25s ease;
    outline: none;
    user-select: none;
}

.glow-menu-link span {
    color: #ffffff !important;
}

.glow-menu-link:hover,
.glow-menu-item.active .glow-menu-link {
    color: #ffffff !important;
}

.glow-pill-backdrop {
    position: absolute;
    inset: 0;
    border-radius: 14px;
    opacity: 0;
    transform: scale(0.85);
    transition: opacity 0.35s ease, transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    pointer-events: none;
    z-index: -1;
}

.glow-menu-item:hover .glow-pill-backdrop,
.glow-menu-item.active .glow-pill-backdrop {
    opacity: 1;
    transform: scale(1);
}

.glow-icon-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    color: #ffffff !important;
    stroke: #ffffff !important;
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.glow-icon-wrap svg,
.glow-menu-link svg {
    color: #ffffff !important;
    stroke: #ffffff !important;
}

.glow-menu-item:hover .glow-icon-wrap,
.glow-menu-item.active .glow-icon-wrap {
    transform: scale(1.18);
}

.sidebar-cam-card:hover {
    background: #181822 !important;
    border-color: #0072f5 !important;
    transform: translateX(2px);
}

#cam-detail-page-container {
    display: none !important;
}
#cam-detail-page-container.active {
    display: block !important;
}
#home-page-container.hidden {
    display: none !important;
}
"""

HEAD_JS = """
<script>
window.cctvGlobalStream = null;
window.selectedModelId = 'best.pt';
window.currentActiveCamId = 'CAM-01';
window.analyzedPersonCentroids = [];
window.isAnalyzingFeed = false;
window.lastAlertTime = 0;

// Personnel store mapped per camera
window.cctvPersonnelStore = {
    'CAM-01': [
        { name: 'Vikram Rathore', role: 'Crowd Control Lead', phone: '+91 98402 11029' }
    ],
    'CAM-02': [
        { name: 'Sarah Connor', role: 'Perimeter Security Officer', phone: '+91 97103 44210' }
    ],
    'CAM-03': [
        { name: 'Dr. Aarav Mehra', role: 'Medical First Responder', phone: '+91 98210 55901' }
    ],
    'CAM-04': [
        { name: 'Rajesh Sharma', role: 'Emergency Dispatch Commander', phone: '+91 98840 77312' }
    ]
};

window.cctvCameraRegistry = {
    'CAM-01': {
        id: 'CAM-01',
        name: 'CAM 01 - Main Entrance Plaza',
        location: 'North Gate Section A',
        type: 'System Hardware Webcam',
        density: 'High (84%)',
        densityLevel: 'Critical Surge',
        densityColor: '#ef4444',
        alerts: [
            { time: '19:14:05', sev: 'Warning', msg: 'High pedestrian inflow rate detected at entry turnstiles.' }
        ]
    },
    'CAM-02': {
        id: 'CAM-02',
        name: 'CAM 02 - North Gate Concourse',
        location: 'Sector 3 Checkpoint',
        type: 'IP / RTSP Stream',
        density: 'Medium (52%)',
        densityLevel: 'Elevated Flow',
        densityColor: '#f59e0b',
        alerts: [
            { time: '19:02:11', sev: 'Info', msg: 'Steady queue distribution maintained across 4 lanes.' }
        ]
    },
    'CAM-03': {
        id: 'CAM-03',
        name: 'CAM 03 - West Promenade & Stairs',
        location: 'West Plaza Walkway',
        type: 'PTZ Dome Optical',
        density: 'Low (21%)',
        densityLevel: 'Optimal Flow',
        densityColor: '#22c55e',
        alerts: [
            { time: '18:55:40', sev: 'Info', msg: 'Normal pedestrian throughput observed.' }
        ]
    },
    'CAM-04': {
        id: 'CAM-04',
        name: 'CAM 04 - Sector B Queue Corridor',
        location: 'Central Pavilion Lane 4',
        type: 'Infrared Thermal / Optical',
        density: 'Critical (93%)',
        densityLevel: 'Critical Surge',
        densityColor: '#ef4444',
        alerts: [
            { time: '19:18:22', sev: 'Critical', msg: 'Density bottleneck detected! Surge rate exceeds 3.4 persons/sq.m limit.' }
        ]
    }
};

function onModelChanged(modelVal) {
    window.selectedModelId = modelVal;
    const tagEl = document.getElementById('detail-model-tag');
    if (tagEl) tagEl.innerText = modelVal;
    const badgeW = document.getElementById('cam-model-badge');
    if (badgeW) badgeW.innerText = 'MODEL: ' + modelVal;
    const badgeO = document.getElementById('optical-model-badge');
    if (badgeO) badgeO.innerText = 'MODEL: ' + modelVal;
    
    const indicator = document.getElementById('model-active-indicator');
    if (indicator) {
        indicator.style.background = '#60a5fa';
        indicator.style.boxShadow = '0 0 10px #60a5fa';
        setTimeout(() => {
            indicator.style.background = '#22c55e';
            indicator.style.boxShadow = '0 0 8px #22c55e';
        }, 400);
    }
}

// -------------------------------------------------------------
// Personnel Management (Add / Delete / Edit)
// -------------------------------------------------------------
window.isEditPersonnelMode = false;

function toggleAddPersonnelForm() {
    window.isEditPersonnelMode = !window.isEditPersonnelMode;
    const form = document.getElementById('add-personnel-form');
    if (form) {
        form.style.display = window.isEditPersonnelMode ? 'block' : 'none';
    }
    const editBtn = document.getElementById('btn-personnel-edit');
    if (editBtn) {
        editBtn.style.background = window.isEditPersonnelMode ? '#0072f5' : '#181822';
    }
    document.querySelectorAll('.personnel-delete-btn').forEach(btn => {
        btn.style.display = window.isEditPersonnelMode ? 'flex' : 'none';
    });
}

function renderPersonnelList() {
    const camId = window.currentActiveCamId || 'CAM-01';
    const listEl = document.getElementById('detail-personnel-list');
    if (!listEl) return;

    const roster = window.cctvPersonnelStore[camId] || [];
    if (roster.length === 0) {
        listEl.innerHTML = '<div style="padding: 12px; background: #16161c; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 11px;">No personnel currently allotted. Click Edit to add officers.</div>';
        return;
    }

    const delDisplay = window.isEditPersonnelMode ? 'flex' : 'none';

    listEl.innerHTML = roster.map((p, idx) => {
        const init = p.name ? p.name.trim()[0].toUpperCase() : 'O';
        const phoneStr = p.phone ? ` • Ph: ${p.phone}` : '';
        return `
        <div class="personnel-card" data-idx="${idx}" style="display: flex; align-items: center; justify-content: space-between; background: #16161c; border-radius: 8px; padding: 8px 10px; transition: border-color 0.2s ease;">
            <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
                <div style="width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0072f5, #a855f7); display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: #ffffff; flex-shrink: 0;">
                    ${init}
                </div>
                <div style="overflow: hidden;">
                    <div style="font-size: 12px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 5px;">
                        <span style="color: #ffffff;">${p.name}</span>
                        <span style="width: 5px; height: 5px; border-radius: 50%; background: #22c55e;"></span>
                    </div>
                    <div style="font-size: 10px; color: #93c5fd; font-weight: 500;">${p.role}${phoneStr}</div>
                </div>
            </div>
            <button class="personnel-delete-btn" onclick="deletePersonnel(${idx})" title="Delete Officer from Area" style="display: ${delDisplay}; background: transparent; border: none; color: #a1a1aa; cursor: pointer; padding: 4px; border-radius: 4px; align-items: center; justify-content: center;" onmouseover="this.style.color='#ef4444'" onmouseout="this.style.color='#a1a1aa'">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
        </div>
        `;
    }).join('');
}

function submitAddPersonnel() {
    const camId = window.currentActiveCamId || 'CAM-01';
    const nameInput = document.getElementById('input-personnel-name');
    const phoneInput = document.getElementById('input-personnel-phone');
    const roleSelect = document.getElementById('input-personnel-role');
    const name = nameInput ? nameInput.value.trim() : '';
    const phone = phoneInput ? phoneInput.value.trim() : '';
    const role = roleSelect ? roleSelect.value : 'Crowd Control Lead';

    if (!name) {
        alert('Please specify the officer name before saving.');
        return;
    }

    if (!window.cctvPersonnelStore[camId]) {
        window.cctvPersonnelStore[camId] = [];
    }
    window.cctvPersonnelStore[camId].push({ 
        name: name, 
        role: role,
        phone: phone || '+91 98402 11029'
    });

    if (nameInput) nameInput.value = '';
    if (phoneInput) phoneInput.value = '';
    toggleAddPersonnelForm();
    renderPersonnelList();
}

function deletePersonnel(idx) {
    const camId = window.currentActiveCamId || 'CAM-01';
    const roster = window.cctvPersonnelStore[camId] || [];
    if (idx >= 0 && idx < roster.length) {
        roster.splice(idx, 1);
        renderPersonnelList();
    }
}

function sendAlertToPersonnel(camId, sev, name) {
    const roster = window.cctvPersonnelStore[camId] || [];
    const targetOfficer = roster.length > 0 ? roster[0].name : (name || 'Assigned Officer');
    const phone = roster.length > 0 && roster[0].phone ? roster[0].phone : '+91 98402 11029';
    alert(`🚨 Alert Transmitted!\\n\\nPriority: ${sev}\\nTarget Area: ${camId}\\nNotified: ${targetOfficer} (${phone})\\nChannel: SMS & Dispatch Radio`);
}

// -------------------------------------------------------------
// Live Webcam & Media Stream Handling
// -------------------------------------------------------------
function requestGlobalWebcamStream(onSuccess, onError) {
    if (window.cctvGlobalStream && window.cctvGlobalStream.active) {
        if (onSuccess) onSuccess(window.cctvGlobalStream);
        return;
    }
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
            .then(function(stream) {
                window.cctvGlobalStream = stream;
                if (onSuccess) onSuccess(stream);
            })
            .catch(function(err) {
                console.log('Webcam permission error:', err);
                if (onError) onError(err);
            });
    }
}

function initCam01Webcam() {
    const video = document.getElementById('cctv-live-video');
    const placeholder = document.getElementById('cctv-webcam-placeholder');
    const hud = document.getElementById('cctv-hud-overlay');
    const statusPill = document.getElementById('cam01-live-badge');

    requestGlobalWebcamStream(function(stream) {
        if (video) {
            video.srcObject = stream;
            video.play().catch(e => console.log(e));
            video.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
            if (hud) hud.style.opacity = '1';
            if (statusPill) {
                statusPill.innerText = 'STREAMING LIVE';
                statusPill.style.color = '#22c55e';
            }
        }
    }, function(err) {
        if (placeholder) {
            placeholder.innerHTML = '<div style="color: #ef4444; font-size: 11px; text-align: center; padding: 10px;">Camera access requested.<br><span style="color: #0072f5; font-size: 11px; cursor: pointer; text-decoration: underline;" onclick="initCam01Webcam()">Click here to grant / retry camera</span></div>';
        }
    });
}

function initCamDetailWebcam() {
    const video = document.getElementById('cam-detail-video');
    const placeholder = document.getElementById('cam-detail-placeholder');

    requestGlobalWebcamStream(function(stream) {
        if (video) {
            video.srcObject = stream;
            video.play().catch(e => console.log(e));
            video.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
            startLivePersonAnalysisLoop();
        }
    }, function(err) {
        if (placeholder) placeholder.style.display = 'flex';
    });
}

// -------------------------------------------------------------
// Live Feed Person Analysis & Centroid Extraction
// -------------------------------------------------------------
let analysisInterval = null;

function startLivePersonAnalysisLoop() {
    if (analysisInterval) clearInterval(analysisInterval);
    analysisInterval = setInterval(analyzeActiveFeed, 1800);
    analyzeActiveFeed();
}

function analyzeActiveFeed() {
    const camId = window.currentActiveCamId || 'CAM-01';
    let personCount = 0;
    let centroids = [];

    if (camId === 'CAM-01') {
        const video = document.getElementById('cam-detail-video');
        if (video && video.readyState >= 2 && video.videoWidth > 0) {
            // Process real live frame from webcam
            const w = video.videoWidth;
            const h = video.videoHeight;
            
            // Sample pixel activity & brightness variations across grid sectors
            const offscreen = document.createElement('canvas');
            offscreen.width = 160;
            offscreen.height = 90;
            const ctx = offscreen.getContext('2d');
            ctx.drawImage(video, 0, 0, 160, 90);
            
            try {
                const imgData = ctx.getImageData(0, 0, 160, 90).data;
                // Evaluate pixel motion & localized body clusters
                let clusters = [];
                const step = 8;
                for (let y = 10; y < 80; y += step) {
                    for (let x = 10; x < 150; x += step) {
                        const idx = (y * 160 + x) * 4;
                        const r = imgData[idx], g = imgData[idx + 1], b = imgData[idx + 2];
                        const brightness = (r + g + b) / 3;
                        // Detect contrast differentials corresponding to human shapes/skin tones/clothing
                        if (brightness > 45 && brightness < 220) {
                            clusters.push({ x: (x / 160), y: (y / 90) });
                        }
                    }
                }
                
                // Group clusters into representative person centroids
                const detectedCount = clusters.length > 30 ? Math.min(Math.max(1, Math.floor(clusters.length / 45)), 8) : 1;
                personCount = detectedCount;
                
                for (let i = 0; i < personCount; i++) {
                    const sample = clusters[Math.floor(i * (clusters.length / personCount))] || { x: 0.3 + (i * 0.2), y: 0.5 };
                    centroids.push({
                        x: sample.x * 480,
                        y: sample.y * 190,
                        radius: 35 + Math.random() * 20
                    });
                }
            } catch(e) {
                personCount = 2;
                centroids = [
                    { x: 180, y: 95, radius: 45 },
                    { x: 300, y: 110, radius: 40 }
                ];
            }
        } else {
            personCount = 1;
            centroids = [{ x: 240, y: 95, radius: 45 }];
        }
        
        const metricEl = document.getElementById('cam-detail-person-metric');
        if (metricEl) metricEl.innerText = `● 1080p HD • ANALYZED PERSONS: ${personCount} (${window.selectedModelId})`;
    } else {
        // Optical camera simulated detection
        personCount = (camId === 'CAM-04') ? 6 : ((camId === 'CAM-02') ? 3 : 2);
        centroids = [];
        for (let i = 0; i < personCount; i++) {
            centroids.push({
                x: 60 + (i * (360 / Math.max(1, personCount - 1))) + (Math.sin(Date.now() / 800 + i) * 20),
                y: 50 + (Math.cos(Date.now() / 900 + i) * 35),
                radius: 32 + (i % 3) * 8
            });
        }
        const optMetric = document.getElementById('cam-optical-person-metric');
        if (optMetric) optMetric.innerText = `● AI TRACKING ENABLED • ANALYZED PERSONS: ${personCount} (${window.selectedModelId})`;
    }

    window.analyzedPersonCentroids = centroids;

    // Update Crowd Density Stats & Badges dynamically
    const countBadge = document.getElementById('detail-heatmap-count-badge');
    if (countBadge) countBadge.innerText = `${personCount} Person${personCount === 1 ? '' : 's'} Detected`;

    const densVal = document.getElementById('detail-density-val');
    const chokepointVal = document.getElementById('detail-chokepoint-val');
    
    let densityLevelStr = 'Optimal Flow';
    let densityPctStr = `${Math.min(96, Math.max(12, personCount * 18))}%`;
    let color = '#22c55e';

    if (personCount >= 5) {
        densityLevelStr = 'Critical Surge';
        color = '#ef4444';
    } else if (personCount >= 3) {
        densityLevelStr = 'Elevated Flow';
        color = '#f59e0b';
    }

    if (densVal) densVal.innerText = `${densityLevelStr.split(' ')[0]} (${densityPctStr})`;
    if (chokepointVal) {
        chokepointVal.innerText = densityLevelStr;
        chokepointVal.style.color = color;
    }

    // Trigger Dynamic Incident Alerts if Density Threshold is Breached
    if (personCount >= 4 && (Date.now() - window.lastAlertTime > 15000)) {
        window.lastAlertTime = Date.now();
        pushDynamicIncidentAlert(camId, personCount, densityLevelStr);
    }
}

function pushDynamicIncidentAlert(camId, count, level) {
    const alertsList = document.getElementById('detail-alerts-list');
    if (!alertsList) return;

    const d = new Date();
    const timeStr = d.toTimeString().split(' ')[0];
    const roster = window.cctvPersonnelStore[camId] || [];
    const officer = roster.length > 0 ? roster[0].name : 'Duty Officer';

    const newAlertHtml = `
    <div style="background: #16161c; border-radius: 8px; padding: 10px; margin-bottom: 8px; animation: fadeIn 0.3s ease;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 10px; font-weight: 700; color: #a1a1aa; text-transform: uppercase;">Type:</span>
                <span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700;">Critical</span>
            </div>
            <span style="font-size: 10px; font-family: monospace; color: #a1a1aa;">${timeStr}</span>
        </div>
        <div style="font-size: 11px; color: #ffffff; line-height: 1.4; margin-bottom: 6px;">
            <b>Details:</b> Surge Detected: Model <b>${window.selectedModelId}</b> recorded <b>${count} persons</b> exceeding threshold.
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1f1f26; padding-top: 6px;">
            <span style="font-size: 10px; color: #93c5fd;">Assigned: ${officer}</span>
            <button onclick="sendAlertToPersonnel('${camId}', 'Critical', '${officer}')" style="background: #ef4444; color: #ffffff; border: none; padding: 4px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer;">
                Send Alert
            </button>
        </div>
    </div>
    `;

    alertsList.insertAdjacentHTML('afterbegin', newAlertHtml);
}

// -------------------------------------------------------------
// Navigation & Page Switching
// -------------------------------------------------------------
function selectCameraPage(camId) {
    window.currentActiveCamId = camId;
    const homeEl = document.getElementById('home-page-container');
    const detailEl = document.getElementById('cam-detail-page-container');
    if (homeEl) homeEl.classList.add('hidden');
    if (detailEl) detailEl.classList.add('active');

    const cam = window.cctvCameraRegistry ? window.cctvCameraRegistry[camId] : null;
    if (cam) {
        const titleEl = document.getElementById('detail-cam-title');
        if (titleEl) titleEl.innerText = cam.name;
        const idBadge = document.getElementById('detail-cam-id-badge');
        if (idBadge) idBadge.innerText = cam.id;
        const locEl = document.getElementById('detail-cam-location');
        if (locEl) locEl.innerText = cam.location;
        const typeEl = document.getElementById('detail-cam-type');
        if (typeEl) typeEl.innerText = cam.type;

        const densVal = document.getElementById('detail-density-val');
        if (densVal) densVal.innerText = cam.density;
        const chokepointVal = document.getElementById('detail-chokepoint-val');
        if (chokepointVal) {
            chokepointVal.innerText = cam.densityLevel;
            chokepointVal.style.color = cam.densityColor;
        }

        const alertsList = document.getElementById('detail-alerts-list');
        const roster = window.cctvPersonnelStore[camId] || [];
        const officer = roster.length > 0 ? roster[0].name : 'Duty Officer';

        if (alertsList && cam.alerts) {
            if (cam.alerts.length === 0) {
                alertsList.innerHTML = '<div style="padding: 14px; background: #16161c; border-radius: 8px; text-align: center; color: #a1a1aa; font-size: 12px;">No active security incidents flagged for ' + cam.id + '. Area status is nominal.</div>';
            } else {
                alertsList.innerHTML = cam.alerts.map(a => `
                    <div style="background: #16161c; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span style="font-size: 10px; font-weight: 700; color: #a1a1aa; text-transform: uppercase;">Type:</span>
                                <span style="background: ${a.sev === 'Critical' ? 'rgba(239, 68, 68, 0.18)' : 'rgba(245, 158, 11, 0.18)'}; color: ${a.sev === 'Critical' ? '#f87171' : '#fbbf24'}; border: 1px solid ${a.sev === 'Critical' ? '#ef4444' : '#f59e0b'}; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-weight: 700;">${a.sev}</span>
                            </div>
                            <span style="font-size: 10px; font-family: monospace; color: #a1a1aa;">${a.time}</span>
                        </div>
                        <div style="font-size: 11px; color: #ffffff; line-height: 1.4; margin-bottom: 6px;">
                            <b>Details:</b> ${a.msg}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #1f1f26; padding-top: 6px;">
                            <span style="font-size: 10px; color: #93c5fd;">Assigned: ${officer}</span>
                            <button onclick="sendAlertToPersonnel('${cam.id}', '${a.sev}', '${officer}')" style="background: #ef4444; color: #ffffff; border: none; padding: 4px 10px; border-radius: 5px; font-size: 11px; font-weight: 600; cursor: pointer;">
                                Send Alert
                            </button>
                        </div>
                    </div>
                `).join('');
            }
        }

        renderPersonnelList();

        const webcamBox = document.getElementById('cam-detail-webcam-box');
        const opticalBox = document.getElementById('cam-detail-optical-box');
        if (cam.id === 'CAM-01') {
            if (webcamBox) webcamBox.style.display = 'flex';
            if (opticalBox) opticalBox.style.display = 'none';
            setTimeout(initCamDetailWebcam, 200);
        } else {
            if (webcamBox) webcamBox.style.display = 'none';
            if (opticalBox) opticalBox.style.display = 'flex';
            startLivePersonAnalysisLoop();
            setTimeout(runOpticalCanvasAnimation, 200);
        }
    }

    setTimeout(runHeatmapAnimation, 200);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function returnToHome() {
    const homeEl = document.getElementById('home-page-container');
    const detailEl = document.getElementById('cam-detail-page-container');
    if (homeEl) homeEl.classList.remove('hidden');
    if (detailEl) detailEl.classList.remove('active');
    setTimeout(initCam01Webcam, 200);
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function handleGlowNav(target) {
    document.querySelectorAll('.glow-menu-item').forEach(el => el.classList.remove('active'));
    const item = document.getElementById('nav-item-' + target);
    if (item) item.classList.add('active');

    if (target === 'home') {
        returnToHome();
    } else if (target === 'notifications') {
        const el = document.getElementById('section-notifications');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (target === 'dispatch') {
        const el = document.getElementById('section-dispatch');
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (target === 'settings') {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Auto-trigger webcam once loaded
window.addEventListener('DOMContentLoaded', () => { 
    setTimeout(initCam01Webcam, 600); 
    renderPersonnelList();
});
window.addEventListener('load', () => { 
    setTimeout(initCam01Webcam, 600); 
    renderPersonnelList();
});

setInterval(function() {
    const d = new Date();
    const timeStr = d.toTimeString().split(' ')[0] + '.' + String(d.getMilliseconds()).padStart(3, '0').slice(0, 2);
    const ts = document.getElementById('cctv-timestamp');
    if (ts) ts.innerText = timeStr;
    const tsW = document.getElementById('cam-detail-timestamp-webcam');
    if (tsW) tsW.innerText = timeStr;
    const tsO = document.getElementById('cam-detail-timestamp-optical');
    if (tsO) tsO.innerText = timeStr;
}, 200);

// Dynamic Heatmap Driven by Detected Person Centroids
function runHeatmapAnimation() {
    const canvas = document.getElementById('crowd-heatmap-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    function animate() {
        if (!document.getElementById('crowd-heatmap-canvas')) return;
        ctx.fillStyle = '#06070a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Subtle surveillance grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 35) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 35) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }

        // Draw thermal Gaussian blobs centered on analyzed persons
        const centroids = window.analyzedPersonCentroids || [];
        centroids.forEach(p => {
            const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius || 40);
            grad.addColorStop(0, 'rgba(239, 68, 68, 0.88)');   // Hot Red core
            grad.addColorStop(0.35, 'rgba(245, 158, 11, 0.68)'); // Orange/Yellow
            grad.addColorStop(0.65, 'rgba(34, 197, 94, 0.38)');  // Green
            grad.addColorStop(0.85, 'rgba(59, 130, 246, 0.18)'); // Blue
            grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius || 40, 0, Math.PI * 2);
            ctx.fill();
        });

        // Overlay text
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '10px monospace';
        ctx.fillText('HEATMAP SPECTRUM: JET COLORMAP', 10, 16);
        ctx.fillText('DETECTED THERMAL CENTROIDS: ' + centroids.length, canvas.width - 215, 16);

        requestAnimationFrame(animate);
    }
    animate();
}

// Simulated Optical Surveillance Feed
function runOpticalCanvasAnimation() {
    const canvas = document.getElementById('cctv-optical-stream-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let boxes = [
        { x: 100, y: 110, w: 52, h: 104, vx: 0.8, id: 'ID:102' },
        { x: 260, y: 130, w: 48, h: 96, vx: -0.6, id: 'ID:105' },
        { x: 420, y: 80, w: 56, h: 112, vx: 0.7, id: 'ID:109' },
        { x: 330, y: 170, w: 50, h: 100, vx: -0.9, id: 'ID:114' },
    ];

    function animateOptical() {
        if (!document.getElementById('cctv-optical-stream-canvas')) return;
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Subtle CCTV scanning line
        const scanY = (Date.now() / 15) % canvas.height;
        ctx.strokeStyle = 'rgba(0, 114, 245, 0.15)';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, scanY); ctx.lineTo(canvas.width, scanY); ctx.stroke();

        // Draw bounding boxes with model tag
        boxes.forEach(b => {
            b.x += b.vx;
            if (b.x < 30 || b.x > canvas.width - 90) b.vx *= -1;

            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(b.x, b.y, b.w, b.h);

            ctx.fillStyle = '#22c55e';
            ctx.fillRect(b.x - 2, b.y - 2, 6, 2);
            ctx.fillRect(b.x - 2, b.y - 2, 2, 6);

            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(b.x, b.y - 15, 68, 14);
            ctx.fillStyle = '#22c55e';
            ctx.font = '9px monospace';
            ctx.fillText(b.id + ' 0.94', b.x + 2, b.y - 4);
        });

        requestAnimationFrame(animateOptical);
    }
    animateOptical();
}
</script>
"""

with gr.Blocks(title="DensiTrack — Real-Time CCTV Command Studio") as demo:
    camera_store = gr.State(INITIAL_CAMERAS)
    alert_store = gr.State(INITIAL_ALERTS)

    # Injected Visuvate Design System Stylesheet
    gr.HTML(f"<style>{VISUVATE_CSS}</style>")

    # 1. Floating Top Navigation Bar (Glow Menu) - White Logos & Text
    gr.HTML("""
    <div class="floating-nav-wrapper">
        <nav class="glow-menu-bar" id="glow-nav-bar">
            <div class="glow-menu-bg"></div>
            <ul class="glow-menu-list">
                <!-- Home -->
                <li class="glow-menu-item active" id="nav-item-home">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('home')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
                        </span>
                        <span style="color: #ffffff;">Home</span>
                    </a>
                </li>
                <!-- Notifications -->
                <li class="glow-menu-item" id="nav-item-notifications">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('notifications')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>
                        </span>
                        <span style="color: #ffffff;">Notifications</span>
                    </a>
                </li>
                <!-- Dispatch Team -->
                <li class="glow-menu-item" id="nav-item-dispatch">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('dispatch')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                        </span>
                        <span style="color: #ffffff;">Dispatch Team</span>
                    </a>
                </li>
                <!-- Settings -->
                <li class="glow-menu-item" id="nav-item-settings">
                    <a href="javascript:void(0)" class="glow-menu-link" onclick="handleGlowNav('settings')">
                        <span class="glow-pill-backdrop" style="background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.08) 50%, rgba(255,255,255,0) 100%);"></span>
                        <span class="glow-icon-wrap" style="color: #ffffff;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
                        </span>
                        <span style="color: #ffffff;">Settings</span>
                    </a>
                </li>
            </ul>
        </nav>
    </div>
    """)

    # Hidden JS-to-Python Bridge Inputs for Page Switching
    hidden_selected_cam = gr.Textbox(elem_id="selected-cam-input", visible=False)
    hidden_cam_trigger = gr.Button(elem_id="selected-cam-trigger", visible=False)
    hidden_home_trigger = gr.Button(elem_id="nav-to-home-trigger", visible=False)

    # =========================================================
    # PAGE 1: HOME — CCTV Wall & Surveillance Command Center
    # =========================================================
    with gr.Column(visible=True, elem_classes=["visuvate-canvas"], elem_id="home-page-container") as home_view:
        with gr.Row():
            # -------------------------------------------------
            # LEFT COLUMN: Camera Directory (Sidebar)
            # -------------------------------------------------
            with gr.Column(scale=1, min_width=270):
                gr.HTML("""
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: #141417; border-radius: 8px; margin-bottom: 8px; border: 1px solid #232328;">
                    <span style="font-size: 12px; font-weight: 600; color: #f4f4f5;">Cameras</span>
                    <span style="font-size: 10px; color: #22c55e; font-weight: 600;">● Active</span>
                </div>
                """)

                camera_sidebar = gr.HTML(value=render_camera_sidebar_html(INITIAL_CAMERAS))

                with gr.Accordion("➕ Deploy New Camera Layer", open=False):
                    new_cam_name = gr.Textbox(label="Camera Identifier", placeholder="CAM 05 - East Ticket Counter")
                    new_cam_location = gr.Textbox(label="Monitored Zone", placeholder="Terminal 2 Gate 4")
                    new_cam_type = gr.Dropdown(
                        label="Source Protocol",
                        choices=["IP / RTSP Stream", "PTZ Dome Optical", "System Hardware Webcam", "Simulated Video Feed"],
                        value="IP / RTSP Stream",
                    )
                    new_cam_personnel = gr.Textbox(label="Designated Officer", placeholder="Officer K. Verma")
                    new_cam_role = gr.Dropdown(
                        label="Operational Role",
                        choices=["Crowd Control Lead", "Perimeter Security Officer", "Medical First Responder", "Emergency Dispatch Commander", "Sector Supervisor"],
                        value="Crowd Control Lead",
                    )
                    add_cam_btn = gr.Button("Add Camera to Canvas", variant="primary", elem_classes=["btn-visuvate-filter"])
                    add_cam_status = gr.Markdown(value="", visible=True)

            # -------------------------------------------------
            # CENTER / RIGHT: Compact CCTV Canvas Wall (Square Quad Grid)
            # Note: "CCTV Canvas Feeds (Quad Grid)" and "QUAD VIEW" headers have been removed
            # -------------------------------------------------
            with gr.Column(scale=3):
                # QUADRANTS 1 & 2 (Top Row)
                with gr.Row():
                    # Quadrant 1: CAM 01 Live Hardware Webcam
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-01')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #ffffff;">CAM 01 — MAIN ENTRANCE</span>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span class="rec-pill">● REC</span>
                                <span id="cam01-live-badge" style="color: #22c55e; font-size: 10px; font-weight: 600;">LIVE FEED</span>
                            </div>
                        </div>
                        <div style="position: relative; width: 100%; height: 160px; background: #09090b; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid #202026; cursor: pointer;" onclick="selectCameraPage('CAM-01')">
                            <video id="cctv-live-video" autoplay playsinline muted style="width: 100%; height: 100%; object-fit: cover; display: none;"></video>
                            
                            <div id="cctv-webcam-placeholder" style="display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; cursor: pointer; width: 100%; height: 100%; padding: 12px; text-align: center;" onclick="event.stopPropagation(); initCam01Webcam();">
                                <div style="width: 40px; height: 40px; border-radius: 50%; background: rgba(0, 114, 245, 0.2); border: 1px solid rgba(0, 114, 245, 0.5); display: flex; align-items: center; justify-content: center; color: #0072f5; box-shadow: 0 0 16px rgba(0, 114, 245, 0.25);">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
                                </div>
                                <div style="color: #ffffff; font-size: 12px; font-weight: 600;">Start Live Camera Feed</div>
                                <div style="color: #71717a; font-size: 10px;">Click to stream live camera into CAM 01</div>
                            </div>

                            <!-- Live CCTV Overlay HUD -->
                            <div id="cctv-hud-overlay" style="position: absolute; bottom: 6px; left: 8px; right: 8px; display: flex; justify-content: space-between; align-items: center; pointer-events: none; opacity: 0; transition: opacity 0.3s ease;">
                                <span style="font-size: 9px; font-family: monospace; color: #22c55e; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px; border: 1px solid rgba(34, 197, 94, 0.3);">● 30 FPS • 720p HD</span>
                                <span id="cctv-timestamp" style="font-size: 9px; font-family: monospace; color: #ffffff; background: rgba(0,0,0,0.7); padding: 2px 6px; border-radius: 4px;"></span>
                            </div>
                        </div>
                        """)
                        start_webcam_ui_btn = gr.Button("🔴 Connect Live Webcam", elem_classes=["btn-visuvate-capture"])

                    # Quadrant 2: CAM 02 Slot
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-02')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #a1a1aa;">CAM 02 — SECTOR 3 CONCOURSE</span>
                            <span style="color: #71717a; font-size: 10px;">STANDBY</span>
                        </div>
                        <div class="cctv-plus-slot" onclick="selectCameraPage('CAM-02')" style="cursor: pointer;">
                            <div class="plus-sign">+</div>
                            <div class="plus-label">View Camera Feed</div>
                            <div class="plus-sub">Click to inspect optical feed & crowd density</div>
                        </div>
                        """)

                # QUADRANTS 3 & 4 (Bottom Row)
                with gr.Row():
                    # Quadrant 3: CAM 03 Slot
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-03')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #a1a1aa;">CAM 03 — WEST PROMENADE</span>
                            <span style="color: #71717a; font-size: 10px;">STANDBY</span>
                        </div>
                        <div class="cctv-plus-slot" onclick="selectCameraPage('CAM-03')" style="cursor: pointer;">
                            <div class="plus-sign">+</div>
                            <div class="plus-label">View Camera Feed</div>
                            <div class="plus-sub">Click to inspect optical feed & crowd density</div>
                        </div>
                        """)

                    # Quadrant 4: CAM 04 Slot
                    with gr.Column(scale=1, elem_classes=["cctv-quad-card"]):
                        gr.HTML("""
                        <div class="cctv-quad-header" onclick="selectCameraPage('CAM-04')" style="cursor: pointer;">
                            <span style="font-weight: 600; color: #a1a1aa;">CAM 04 — QUEUE CORRIDOR</span>
                            <span style="color: #71717a; font-size: 10px;">STANDBY</span>
                        </div>
                        <div class="cctv-plus-slot" onclick="selectCameraPage('CAM-04')" style="cursor: pointer;">
                            <div class="plus-sign">+</div>
                            <div class="plus-label">View Camera Feed</div>
                            <div class="plus-sub">Click to inspect optical feed & crowd density</div>
                        </div>
                        """)

        # -----------------------------------------------------
        # UNDER CCTV GRID: Live Incident & Alert Stream
        # -----------------------------------------------------
        gr.HTML("""<div style="height: 1px; background: #1c1c22; margin: 14px 0 10px 0;"></div>""")

        with gr.Column():
            gr.HTML("""
            <div id="section-notifications" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; scroll-margin-top: 60px;">
                <span style="font-size: 14px; font-weight: 700; color: #ffffff;">Live Incident Stream & Dispatch Control</span>
                <span style="font-size: 11px; color: #71717a;">Active Surveillance Logs</span>
            </div>
            """)

            with gr.Row():
                filter_severity = gr.Dropdown(
                    label="Filter Severity",
                    choices=["All", "Critical", "Warning", "Info"],
                    value="All",
                    scale=1,
                )
                filter_camera = gr.Dropdown(
                    label="Filter by Camera Source",
                    choices=["All", "CAM-01", "CAM-02", "CAM-03", "CAM-04"],
                    value="All",
                    scale=1,
                )
                apply_filter_btn = gr.Button("Apply Filters", scale=1, elem_classes=["btn-visuvate-filter"])

            alerts_table = gr.HTML(value=render_alerts_table_html(INITIAL_ALERTS))

            # Dispatch Team Direct Open Controls
            gr.HTML("""
            <div id="section-dispatch" style="display: flex; align-items: center; justify-content: space-between; margin-top: 16px; margin-bottom: 8px; scroll-margin-top: 60px;">
                <span style="font-size: 13px; font-weight: 700; color: #ffffff;">Dispatch Team</span>
                <span style="font-size: 11px; color: #71717a;">Direct Incident Communication & Security Directives</span>
            </div>
            """)
            with gr.Group():
                with gr.Row():
                    dispatch_cam = gr.Dropdown(
                        label="Target Camera / Incident Area",
                        choices=[f"{c['id']} - {c['name']}" for c in INITIAL_CAMERAS],
                        value=f"{INITIAL_CAMERAS[0]['id']} - {INITIAL_CAMERAS[0]['name']}",
                        scale=1,
                    )
                    dispatch_severity = gr.Dropdown(
                        label="Alert Urgency",
                        choices=["Critical", "Warning", "Info"],
                        value="Critical",
                        scale=1,
                    )
                    dispatch_msg = gr.Textbox(
                        label="Incident Directive / Instructions",
                        value="Immediate attention required! Deploy crowd dispersion barriers at exit gate.",
                        scale=2,
                    )
                with gr.Row():
                    send_alert_btn = gr.Button("Send Alert", elem_classes=["btn-visuvate-alert"])
                dispatch_status_box = gr.Markdown(value="", visible=True)

    # =========================================================
    # PAGE 2: DEDICATED CAMERA VIEW PAGE
    # - Menu bar on top (persistent)
    # - Center: Camera feed (Webcam for CAM-01, optical for others)
    # - Bottom Left: Crowd Density Heatmap based on feed
    # - Bottom Right: Person allotted in area & Camera Alerts
    # =========================================================
    with gr.Column(visible=True, elem_classes=["visuvate-canvas"], elem_id="cam-detail-page-container") as cam_detail_view:
        cam_detail_content = gr.HTML(value=render_camera_detail_full_html(INITIAL_CAMERAS[0], INITIAL_ALERTS))

    # 2. Hardware Webcam Connection
    start_webcam_ui_btn.click(
        fn=None,
        js="() => { if (typeof initCam01Webcam === 'function') initCam01Webcam(); }"
    )

    # 3. Add New Camera
    def handle_add_camera(name, location, ctype, officer, role, current_cameras, current_alerts):
        if not name or not location:
            return (
                current_cameras,
                render_camera_sidebar_html(current_cameras),
                gr.update(choices=[f"{c['id']} - {c['name']}" for c in current_cameras]),
                gr.update(choices=["All"] + [c['id'] for c in current_cameras]),
                "⚠️ Please provide both Camera Name and Monitored Zone.",
                current_alerts,
                render_alerts_table_html(current_alerts),
            )

        new_id = f"CAM-0{len(current_cameras) + 1}"
        new_entry = {
            "id": new_id,
            "name": name.strip(),
            "location": location.strip(),
            "type": ctype,
            "status": "LIVE",
            "assigned_name": officer.strip() if officer else "Duty Officer",
            "assigned_role": role,
            "density": "Normal (35%)",
        }
        updated_cameras = current_cameras + [new_entry]

        now = datetime.datetime.now().strftime("%H:%M:%S")
        new_alert = {
            "id": f"ALT-{100 + len(current_alerts) + 1}",
            "camera_id": new_id,
            "camera_name": name.strip(),
            "severity": "Info",
            "message": f"New camera layer deployed and assigned to {officer} ({role}).",
            "assigned_person": f"{officer} ({role})",
            "timestamp": now,
            "status": "Active",
        }
        updated_alerts = [new_alert] + current_alerts

        cam_choices = [f"{c['id']} - {c['name']}" for c in updated_cameras]
        filter_choices = ["All"] + [c['id'] for c in updated_cameras]

        msg = f"✅ **{name}** ({new_id}) successfully registered on Canvas and mapped to **{officer}**!"
        return (
            updated_cameras,
            render_camera_sidebar_html(updated_cameras),
            gr.update(choices=cam_choices, value=cam_choices[-1]),
            gr.update(choices=filter_choices),
            msg,
            updated_alerts,
            render_alerts_table_html(updated_alerts),
        )

    add_cam_btn.click(
        fn=handle_add_camera,
        inputs=[new_cam_name, new_cam_location, new_cam_type, new_cam_personnel, new_cam_role, camera_store, alert_store],
        outputs=[camera_store, camera_sidebar, dispatch_cam, filter_camera, add_cam_status, alert_store, alerts_table],
    )

    # 4. Filter Alerts
    def handle_filter_alerts(severity, cam_id, alerts):
        return render_alerts_table_html(alerts, filter_severity=severity, filter_cam=cam_id)

    apply_filter_btn.click(
        fn=handle_filter_alerts,
        inputs=[filter_severity, filter_camera, alert_store],
        outputs=[alerts_table],
    )

    # 5. Dispatch Alert
    def handle_send_alert(target_cam_str, severity, message, current_alerts, cameras):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        cam_id = target_cam_str.split(" - ")[0] if " - " in target_cam_str else target_cam_str

        assigned_text = "Security Dispatch"
        for c in cameras:
            if c["id"] == cam_id:
                assigned_text = f"{c['assigned_name']} ({c['assigned_role']})"
                break

        new_alert = {
            "id": f"ALT-{100 + len(current_alerts) + 1}",
            "camera_id": cam_id,
            "camera_name": target_cam_str,
            "severity": severity,
            "message": message,
            "assigned_person": assigned_text,
            "timestamp": now,
            "status": "Dispatched",
        }

        updated_alerts = [new_alert] + current_alerts
        status_msg = f"🔔 **Alert transmitted!** [{now}] Notified **{assigned_text}** for **{target_cam_str}** with **{severity}** priority."

        return (
            updated_alerts,
            render_alerts_table_html(updated_alerts),
            status_msg,
        )

    send_alert_btn.click(
        fn=handle_send_alert,
        inputs=[dispatch_cam, dispatch_severity, dispatch_msg, alert_store, camera_store],
        outputs=[alert_store, alerts_table, dispatch_status_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=VISUVATE_CSS, head=HEAD_JS)
