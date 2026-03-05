"""
Generates the EAIGLE AI high-level architecture diagram as PNG and PDF.

Usage:
    python scripts/draw_architecture.py
Outputs:
    docs/architecture.png
    docs/architecture.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

os.makedirs("docs", exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "camera":      "#1A73E8",   # blue
    "ingestion":   "#0D47A1",   # dark blue
    "shm":         "#37474F",   # dark grey (shared memory)
    "redis":       "#CC0000",   # Redis red
    "preprocess":  "#2E7D32",   # green
    "inference":   "#6A1B9A",   # purple
    "infsvc":      "#7B1FA2",   # lighter purple
    "aggregation": "#E65100",   # orange
    "output":      "#00695C",   # teal
    "arrow":       "#455A64",   # slate
    "bg":          "#FAFAFA",
    "white":       "#FFFFFF",
    "text_dark":   "#212121",
    "text_light":  "#FFFFFF",
    "border":      "#B0BEC5",
    "accent":      "#FFD600",
}

FIG_W, FIG_H = 22, 28
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=C["bg"])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])

# ── Helpers ──────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, color, radius=0.35, alpha=1.0, zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="none",
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, fontsize=11, color=C["text_light"],
          bold=False, zorder=5, ha="center", va="center", wrap=False):
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, color=color,
            fontweight=weight, ha=ha, va=va, zorder=zorder,
            wrap=wrap)


def arrow(ax, x1, y1, x2, y2, color=C["arrow"], lw=2.0,
          label_text="", zorder=4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.35,head_length=0.3",
            color=color, lw=lw,
        ),
        zorder=zorder,
    )
    if label_text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label_text, fontsize=7.5,
                color=color, ha="left", va="center", zorder=5,
                style="italic")


def section_bg(ax, x, y, w, h, color, label_text, zorder=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.5",
        facecolor=color, edgecolor=C["border"],
        alpha=0.08, zorder=zorder, linewidth=1.2, linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(x + 0.25, y + h - 0.28, label_text,
            fontsize=8.5, color=color, fontweight="bold",
            va="top", ha="left", zorder=zorder + 1, alpha=0.9)


# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
ax.text(FIG_W / 2, 27.4,
        "EAIGLE AI — Multi-Camera Computer Vision Pipeline",
        fontsize=18, color=C["text_dark"], fontweight="bold",
        ha="center", va="center", zorder=6)
ax.text(FIG_W / 2, 26.9,
        "High-Level Architecture",
        fontsize=12, color="#607D8B", ha="center", va="center", zorder=6)

# Divider
ax.axhline(26.6, xmin=0.02, xmax=0.98,
           color=C["border"], linewidth=1.2, zorder=3)

# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — CAMERAS  (y ≈ 24.5)
# ════════════════════════════════════════════════════════════════════════════
cam_y = 24.0
cam_w, cam_h = 2.6, 1.3
cam_gap = 0.45
cam_labels = ["Cam 01", "Cam 02", "Cam 03", "···", "Cam 50"]
cam_xs = [1.2, 4.45, 7.7, 11.0, 14.25]

for i, (cx, cl) in enumerate(zip(cam_xs, cam_labels)):
    is_dots = cl == "···"
    c = C["border"] if is_dots else C["camera"]
    box(ax, cx, cam_y, cam_w, cam_h, c, radius=0.3)
    label(ax, cx + cam_w / 2, cam_y + cam_h / 2, cl,
          fontsize=11 if not is_dots else 18,
          color=C["text_light"] if not is_dots else C["text_dark"],
          bold=not is_dots)
    if not is_dots:
        label(ax, cx + cam_w / 2, cam_y + 0.22, "RTSP", fontsize=8,
              color="#BBDEFB")

label(ax, FIG_W / 2, cam_y + cam_h + 0.35,
      "Up to 50 concurrent RTSP streams (H.264 / H.265)",
      fontsize=9, color="#607D8B", bold=False)

# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — INGESTION  (y ≈ 21)
# ════════════════════════════════════════════════════════════════════════════
ing_y = 20.7
ing_bg_x = 0.8
section_bg(ax, ing_bg_x, ing_y - 0.1, FIG_W - 1.6, 2.5,
           C["ingestion"], "INGESTION LAYER")

# OpenCV backend
ocv_x, ocv_w, ocv_h = 1.5, 8.0, 1.9
box(ax, ocv_x, ing_y + 0.2, ocv_w, ocv_h, C["ingestion"], radius=0.3)
label(ax, ocv_x + ocv_w / 2, ing_y + 1.35,
      "OpenCV Backend  (CPU)", fontsize=11, bold=True)
label(ax, ocv_x + ocv_w / 2, ing_y + 0.88,
      "asyncio + ThreadPoolExecutor", fontsize=9)
label(ax, ocv_x + ocv_w / 2, ing_y + 0.50,
      "cv2.VideoCapture  ·  auto-reconnect  ·  back-pressure", fontsize=8.5,
      color="#90CAF9")

# DeepStream backend
ds_x = 11.2
box(ax, ds_x, ing_y + 0.2, ocv_w, ocv_h, "#311B92", radius=0.3)
label(ax, ds_x + ocv_w / 2, ing_y + 1.35,
      "DeepStream Backend  (GPU)", fontsize=11, bold=True)
label(ax, ds_x + ocv_w / 2, ing_y + 0.88,
      "GStreamer  ·  GLib main loop  ·  nvurisrcbin", fontsize=9)
label(ax, ds_x + ocv_w / 2, ing_y + 0.50,
      "nvstreammux  ·  NVDEC HW decode  ·  pad probe", fontsize=8.5,
      color="#CE93D8")

# OR label
box(ax, 9.7, ing_y + 0.75, 1.4, 0.8, "#607D8B", radius=0.25)
label(ax, 10.4, ing_y + 1.15, "OR", fontsize=12, bold=True)
label(ax, 10.4, ing_y + 0.85, "config-\nselected", fontsize=7,
      color="#CFD8DC")

# Arrows cameras → ingestion (centre)
for cx in [cam_xs[0] + cam_w / 2, cam_xs[2] + cam_w / 2,
           cam_xs[4] + cam_w / 2]:
    arrow(ax, cx, cam_y, 11.1, ing_y + 2.1, C["camera"], lw=1.4)

# ════════════════════════════════════════════════════════════════════════════
# ROW 3 — SHARED MEMORY + REDIS  (y ≈ 18.5)
# ════════════════════════════════════════════════════════════════════════════
tr_y = 18.3

# /dev/shm box
shm_x, shm_w, shm_h = 1.5, 7.5, 1.4
box(ax, shm_x, tr_y, shm_w, shm_h, C["shm"], radius=0.3)
label(ax, shm_x + shm_w / 2, tr_y + 0.85,
      "/dev/shm  (POSIX Shared Memory)", fontsize=10.5, bold=True)
label(ax, shm_x + shm_w / 2, tr_y + 0.40,
      "Pixel data — 6 MB/frame — zero serialisation overhead", fontsize=8.5,
      color="#B0BEC5")

# Redis Streams box
red_x, red_w = 10.5, 10.0
box(ax, red_x, tr_y, red_w, shm_h, C["redis"], radius=0.3)
label(ax, red_x + red_w / 2, tr_y + 0.85,
      "Redis Streams  (raw_frames:{cam_id} × 50)", fontsize=10.5, bold=True)
label(ax, red_x + red_w / 2, tr_y + 0.40,
      "FrameMetadata pointer  ·  ~200 bytes  ·  XREADGROUP consumer groups",
      fontsize=8.5, color="#FFCDD2")

# Arrows ingestion → shm / redis
arrow(ax, 5.5, ing_y + 0.2, 5.5, tr_y + shm_h, C["shm"], lw=2,
      label_text="write pixels")
arrow(ax, 15.5, ing_y + 0.2, 15.5, tr_y + shm_h, C["redis"], lw=2,
      label_text="publish pointer")

# ════════════════════════════════════════════════════════════════════════════
# ROW 4 — PREPROCESSING  (y ≈ 15.8)
# ════════════════════════════════════════════════════════════════════════════
pp_y = 15.6
section_bg(ax, ing_bg_x, pp_y - 0.1, FIG_W - 1.6, 2.5,
           C["preprocess"], "PREPROCESSING LAYER")

pp_x, pp_w, pp_h = 1.5, 19.2, 1.9
box(ax, pp_x, pp_y + 0.2, pp_w, pp_h, C["preprocess"], radius=0.3)
label(ax, pp_x + pp_w / 2, pp_y + 1.35,
      "PreprocessingWorkerPool  —  ProcessPoolExecutor  (N workers, bypasses GIL)",
      fontsize=11, bold=True)
label(ax, pp_x + pp_w / 2, pp_y + 0.85,
      "Consumes raw_frames via XREADGROUP  ·  reads pixel data from /dev/shm",
      fontsize=9)

# Op pipeline boxes inside
ops = ["BGR → RGB", "Resize\n640×640", "Gaussian\nDenoise", "ImageNet\nNormalize"]
op_colors = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047"]
op_w = 3.5
op_start = 3.2
for i, (op, oc) in enumerate(zip(ops, op_colors)):
    ox = op_start + i * (op_w + 0.55)
    box(ax, ox, pp_y + 0.28, op_w, 0.72, oc, radius=0.2)
    label(ax, ox + op_w / 2, pp_y + 0.64, op, fontsize=8.5)
    if i < len(ops) - 1:
        ax.annotate("", xy=(ox + op_w + 0.55, pp_y + 0.64),
                    xytext=(ox + op_w, pp_y + 0.64),
                    arrowprops=dict(arrowstyle="->", color="#A5D6A7", lw=1.2),
                    zorder=6)

# Arrows shm/redis → preprocessing
arrow(ax, 5.25, tr_y, 5.25, pp_y + 2.1, C["shm"], lw=2,
      label_text="read pixels")
arrow(ax, 15.5, tr_y, 15.5, pp_y + 2.1, C["redis"], lw=2,
      label_text="XREADGROUP")

# ════════════════════════════════════════════════════════════════════════════
# ROW 5 — REDIS preprocessed_frames  (y ≈ 13.5)
# ════════════════════════════════════════════════════════════════════════════
pr_y = 13.4
box(ax, ing_bg_x, pr_y, FIG_W - 1.6, 1.2, C["redis"], radius=0.3)
label(ax, FIG_W / 2, pr_y + 0.75,
      "Redis Streams  —  preprocessed_frames", fontsize=11, bold=True)
label(ax, FIG_W / 2, pr_y + 0.33,
      "Single stream  ·  consumer group 'inference_workers'  ·  XADD / XREADGROUP / XACK",
      fontsize=9, color="#FFCDD2")

arrow(ax, FIG_W / 2, pp_y + 0.2, FIG_W / 2, pr_y + 1.2,
      C["preprocess"], lw=2.5, label_text="publish preprocessed pointer")

# ════════════════════════════════════════════════════════════════════════════
# ROW 6 — INFERENCE DISPATCH + INFERENCE SERVICE  (y ≈ 10.5)
# ════════════════════════════════════════════════════════════════════════════
id_y = 10.3
section_bg(ax, ing_bg_x, id_y - 0.15, FIG_W - 1.6, 3.0,
           C["inference"], "INFERENCE LAYER")

# Dispatcher + Batcher
disp_x, disp_w, disp_h = 1.5, 9.5, 2.55
box(ax, disp_x, id_y + 0.1, disp_w, disp_h, C["inference"], radius=0.3)
label(ax, disp_x + disp_w / 2, id_y + 2.2,
      "InferenceDispatcher", fontsize=11, bold=True)
label(ax, disp_x + disp_w / 2, id_y + 1.70,
      "Reads frames from /dev/shm", fontsize=8.5, color="#CE93D8")

# DynamicBatcher
db_bx, db_bw, db_bh = disp_x + 0.4, 8.6, 0.95
box(ax, db_bx, id_y + 0.35, db_bw, db_bh, "#4A148C", radius=0.2)
label(ax, db_bx + db_bw / 2, id_y + 0.95,
      "DynamicBatcher", fontsize=10, bold=True)
label(ax, db_bx + db_bw / 2, id_y + 0.58,
      "size ≤ 8 frames  OR  time ≤ 50 ms  →  flush", fontsize=8.5,
      color="#EDE7F6")

# Inference Service
svc_x, svc_w, svc_h = 12.5, 8.7, 2.55
box(ax, svc_x, id_y + 0.1, svc_w, svc_h, "#7B1FA2", radius=0.3)
label(ax, svc_x + svc_w / 2, id_y + 2.2,
      "Inference Service  (FastAPI microservice)", fontsize=11, bold=True)
label(ax, svc_x + svc_w / 2, id_y + 1.75,
      "Independent process / container", fontsize=8.5, color="#CE93D8")

# Stage A box
sa_bx, sa_bw, sa_bh = svc_x + 0.35, 3.8, 0.95
box(ax, sa_bx, id_y + 0.95, sa_bw, sa_bh, "#6A1B9A", radius=0.2)
label(ax, sa_bx + sa_bw / 2, id_y + 1.52,
      "Stage A: Object Detection", fontsize=8.5, bold=True)
label(ax, sa_bx + sa_bw / 2, id_y + 1.12,
      "YOLO / DETR / Faster R-CNN", fontsize=7.5, color="#EDE7F6")

# Stage B box
sb_bx = svc_x + 0.35 + sa_bw + 0.45
box(ax, sb_bx, id_y + 0.95, sa_bw, sa_bh, "#4A148C", radius=0.2)
label(ax, sb_bx + sa_bw / 2, id_y + 1.52,
      "Stage B: Crop Classifier / OCR", fontsize=8.5, bold=True)
label(ax, sb_bx + sa_bw / 2, id_y + 1.12,
      "Plate → 'ABC123'  ·  Badge → 'ID-4821'", fontsize=7.5,
      color="#EDE7F6")

# A → B arrow inside service
ax.annotate("", xy=(sb_bx, id_y + 1.43), xytext=(sa_bx + sa_bw, id_y + 1.43),
            arrowprops=dict(arrowstyle="->", color="#CE93D8", lw=1.4), zorder=6)

# label boxes inside service
label(ax, svc_x + svc_w / 2, id_y + 0.48,
      "POST /detect/primary  ·  POST /detect/secondary", fontsize=8,
      color="#CE93D8")
box(ax, svc_x + 0.35, id_y + 0.12, svc_w - 0.7, 0.5, "#4A148C", radius=0.15)
label(ax, svc_x + svc_w / 2, id_y + 0.37,
      "GPU   ·   TensorRT   ·   Hardware batch inference", fontsize=8,
      color="#EDE7F6")

# HTTP arrow dispatcher → service
arrow(ax, disp_x + disp_w, id_y + 1.35, svc_x, id_y + 1.35,
      C["accent"], lw=2.5, label_text="HTTP POST (batch, base64)")

# Arrows redis → dispatcher
arrow(ax, FIG_W / 2, pr_y, 6.5, id_y + 2.65, C["redis"], lw=2,
      label_text="XREADGROUP")

# ════════════════════════════════════════════════════════════════════════════
# ROW 7 — REDIS inference_results  (y ≈ 8.3)
# ════════════════════════════════════════════════════════════════════════════
ir_y = 8.1
box(ax, ing_bg_x, ir_y, FIG_W - 1.6, 1.2, C["redis"], radius=0.3)
label(ax, FIG_W / 2, ir_y + 0.75,
      "Redis Streams  —  inference_results", fontsize=11, bold=True)
label(ax, FIG_W / 2, ir_y + 0.33,
      "StageResult JSON  ·  consumer group 'aggregation_workers'",
      fontsize=9, color="#FFCDD2")

arrow(ax, 6.5, id_y + 0.1, 6.5, ir_y + 1.2,
      C["inference"], lw=2.5, label_text="publish StageResults")

# ════════════════════════════════════════════════════════════════════════════
# ROW 8 — AGGREGATION  (y ≈ 5.2)
# ════════════════════════════════════════════════════════════════════════════
ag_y = 5.0
section_bg(ax, ing_bg_x, ag_y - 0.15, FIG_W - 1.6, 3.0,
           C["aggregation"], "AGGREGATION LAYER")

# ResultCollector
rc_x, rc_w, rc_h = 1.5, 5.5, 2.55
box(ax, rc_x, ag_y + 0.1, rc_w, rc_h, "#BF360C", radius=0.3)
label(ax, rc_x + rc_w / 2, ag_y + 1.9,
      "ResultCollector", fontsize=11, bold=True)
label(ax, rc_x + rc_w / 2, ag_y + 1.45,
      "Groups StageResults by frame_id", fontsize=8.5, color="#FFCCBC")
label(ax, rc_x + rc_w / 2, ag_y + 1.05,
      "Waits for all stages  ·  2s timeout flush", fontsize=8,
      color="#FFCCBC")
box(ax, rc_x + 0.35, ag_y + 0.2, rc_w - 0.7, 0.7, "#7F2700", radius=0.2)
label(ax, rc_x + rc_w / 2, ag_y + 0.55,
      "FrameAccumulator  per  frame_id", fontsize=8, color="#FFCCBC")

# HypothesisFusion
hf_x, hf_w, hf_h = 8.0, 6.5, 2.55
box(ax, hf_x, ag_y + 0.1, hf_w, hf_h, C["aggregation"], radius=0.3)
label(ax, hf_x + hf_w / 2, ag_y + 2.2,
      "HypothesisFusion", fontsize=11, bold=True)

fus_ops = [("SpatialNMS", "IoU > 0.45\n→ keep best"),
           ("ConfidenceVoting", "score < 0.40\n→ discard"),
           ("TemporalSmoother", "5-frame window\nmin 2 appearances")]
fop_w = 1.75
for i, (fname, fdesc) in enumerate(fus_ops):
    fx = hf_x + 0.35 + i * (fop_w + 0.3)
    box(ax, fx, ag_y + 0.2, fop_w, 1.45, "#E64A19", radius=0.2)
    label(ax, fx + fop_w / 2, ag_y + 1.38, fname, fontsize=7.5, bold=True)
    label(ax, fx + fop_w / 2, ag_y + 0.78, fdesc, fontsize=7,
          color="#FFCCBC")
    if i < 2:
        ax.annotate("", xy=(fx + fop_w + 0.3, ag_y + 0.93),
                    xytext=(fx + fop_w, ag_y + 0.93),
                    arrowprops=dict(arrowstyle="->", color="#FFAB91", lw=1.2),
                    zorder=6)

# HypothesisStore
hs_x, hs_w, hs_h = 15.7, 5.1, 2.55
box(ax, hs_x, ag_y + 0.1, hs_w, hs_h, "#004D40", radius=0.3)
label(ax, hs_x + hs_w / 2, ag_y + 2.2,
      "HypothesisStore", fontsize=11, bold=True)
label(ax, hs_x + hs_w / 2, ag_y + 1.68,
      "HASH hypothesis:{id}  (60s TTL)", fontsize=8, color="#B2DFDB")
label(ax, hs_x + hs_w / 2, ag_y + 1.28,
      "LIST camera_hypotheses:{cam} (100)", fontsize=8, color="#B2DFDB")
label(ax, hs_x + hs_w / 2, ag_y + 0.88,
      "PUBLISH  →  hypotheses  channel", fontsize=8, color="#B2DFDB")
box(ax, hs_x + 0.35, ag_y + 0.18, hs_w - 0.7, 0.55, "#00251A", radius=0.2)
label(ax, hs_x + hs_w / 2, ag_y + 0.46,
      "Redis", fontsize=9, bold=True, color="#80CBC4")

# Arrows within aggregation
arrow(ax, rc_x + rc_w, ag_y + 1.35, hf_x, ag_y + 1.35, C["aggregation"], lw=2)
arrow(ax, hf_x + hf_w, ag_y + 1.35, hs_x, ag_y + 1.35, C["aggregation"], lw=2)

# Redis → ResultCollector
arrow(ax, FIG_W / 2, ir_y, 4.25, ag_y + 2.65, C["redis"], lw=2,
      label_text="XREADGROUP")

# ════════════════════════════════════════════════════════════════════════════
# ROW 9 — EVENT DESCRIPTION  (y ≈ 3.2)
# ════════════════════════════════════════════════════════════════════════════
ev_y = 3.0
box(ax, ing_bg_x, ev_y, FIG_W - 1.6, 1.2, "#004D40", radius=0.3)
label(ax, FIG_W / 2, ev_y + 0.85,
      "\"Detected at Gate 3 [cam_03]: Vehicle 'ABC123' (license_plate, 91%);  Person (75%)\"",
      fontsize=10.5, bold=True, color=C["accent"])
label(ax, FIG_W / 2, ev_y + 0.35,
      "Human-readable Hypothesis  ·  confidence  ·  camera zone  ·  OCR text",
      fontsize=9, color="#B2DFDB")

arrow(ax, 18.25, ag_y + 0.1, 18.25, ev_y + 1.2, C["output"], lw=2.5,
      label_text="publish event")

# ════════════════════════════════════════════════════════════════════════════
# ROW 10 — DOWNSTREAM CONSUMERS  (y ≈ 1.0)
# ════════════════════════════════════════════════════════════════════════════
ds_y = 0.6
consumers = [
    ("Dashboard /\nAPI", "#01579B"),
    ("Alert\nService", "#B71C1C"),
    ("Storage /\nArchive", "#1B5E20"),
    ("Analytics\n& BI", "#4A148C"),
    ("Prometheus\nMetrics", "#E65100"),
]
con_w = 3.0
con_start = (FIG_W - len(consumers) * con_w - (len(consumers) - 1) * 0.55) / 2
for i, (clabel, ccolor) in enumerate(consumers):
    cx = con_start + i * (con_w + 0.55)
    box(ax, cx, ds_y, con_w, 1.3, ccolor, radius=0.3)
    label(ax, cx + con_w / 2, ds_y + 0.65, clabel, fontsize=9.5)

arrow(ax, FIG_W / 2, ev_y, FIG_W / 2, ds_y + 1.3, C["output"], lw=2.5,
      label_text="Redis PubSub  /  Webhook")

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
legend_x, legend_y = 0.85, 12.6
legend_items = [
    (C["camera"],    "RTSP Camera"),
    (C["ingestion"], "Ingestion (OpenCV)"),
    ("#311B92",      "Ingestion (DeepStream/GPU)"),
    (C["shm"],       "Shared Memory (/dev/shm)"),
    (C["redis"],     "Redis Streams / PubSub"),
    (C["preprocess"],"Preprocessing (CPU ProcessPool)"),
    (C["inference"], "Inference Dispatcher"),
    ("#7B1FA2",      "Inference Service"),
    (C["aggregation"],"Aggregation Layer"),
    (C["output"],    "Hypothesis / Output"),
]
ax.text(legend_x, legend_y + 0.3,
        "Legend", fontsize=9, fontweight="bold", color=C["text_dark"])
for i, (lc, lt) in enumerate(legend_items):
    ly = legend_y - 0.05 - i * 0.52
    box(ax, legend_x, ly, 0.5, 0.36, lc, radius=0.1)
    ax.text(legend_x + 0.65, ly + 0.18, lt,
            fontsize=7.5, va="center", color=C["text_dark"])

# Border frame
border = FancyBboxPatch(
    (0.1, 0.1), FIG_W - 0.2, FIG_H - 0.2,
    boxstyle="round,pad=0,rounding_size=0.3",
    facecolor="none", edgecolor=C["border"],
    linewidth=1.5, zorder=0,
)
ax.add_patch(border)

# ── Save ──────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0)

png_path = "docs/architecture.png"
pdf_path = "docs/architecture.pdf"
plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=C["bg"])
plt.savefig(pdf_path, bbox_inches="tight", facecolor=C["bg"])
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
