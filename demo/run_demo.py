#!/usr/bin/env python3
"""
run_demo.py

A Cosmos Cookoff-friendly demo runner:
- Loads a video
- Samples short time windows (clips) from it
- For each clip, calls a "Cosmos Reason 2" function (stub or real API)
- Writes JSON outputs per window
- Optionally writes an overlay video with the detected state/actions

Usage:
  python demo/run_demo.py --video demo/input_videos/angry.mp4 --outdir demo/outputs --overlay

Notes:
- This script includes TWO modes:
  1) STUB mode (default): no external API needed, generates plausible outputs
  2) API mode: implement `call_cosmos_reason_api(...)` with your Cosmos endpoint

You can ship this as a runnable skeleton.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception as e:
    cv2 = None  # type: ignore


# ----------------------------
# Data models
# ----------------------------

@dataclass
class RecommendedAction:
    action: str
    value: Any

@dataclass
class CosmosResult:
    emotion: str
    confidence: float
    evidence: List[str]
    interaction_state: str
    risk_level: str
    recommended_robot_actions: List[RecommendedAction]

    def to_json(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion,
            "confidence": round(float(self.confidence), 3),
            "evidence": self.evidence,
            "interaction_state": self.interaction_state,
            "risk_level": self.risk_level,
            "recommended_robot_actions": [
                {"action": a.action, "value": a.value} for a in self.recommended_robot_actions
            ],
        }


# ----------------------------
# Helpers
# ----------------------------

EMOTION_TO_STATE = {
    "angry": "deescalate",
    "happy": "engage",
    "sad": "support",
    "frustrated": "assist",
    "neutral": "neutral",
}

STATE_DEFAULT_ACTIONS = {
    "deescalate": [
        RecommendedAction("increase_distance_m", 0.5),
        RecommendedAction("speech_rate", "slow"),
        RecommendedAction("tone", "calm"),
    ],
    "engage": [
        RecommendedAction("increase_gesture_energy", "medium"),
        RecommendedAction("speech_rate", "normal"),
        RecommendedAction("tone", "friendly"),
    ],
    "support": [
        RecommendedAction("reduce_motion_speed", "slow"),
        RecommendedAction("speech_rate", "slow"),
        RecommendedAction("tone", "gentle"),
    ],
    "assist": [
        RecommendedAction("offer_step_by_step", True),
        RecommendedAction("speech_rate", "clear"),
        RecommendedAction("tone", "helpful"),
    ],
    "neutral": [
        RecommendedAction("speech_rate", "normal"),
        RecommendedAction("tone", "neutral"),
    ],
}

EMOTION_EVIDENCE_BANK = {
    "angry": ["tense posture", "sharp gestures", "brows lowered", "loud voice cues (visual)"],
    "happy": ["smiling", "relaxed shoulders", "open posture", "light gestures"],
    "sad": ["downward gaze", "slumped posture", "slow movement", "quiet demeanor cues (visual)"],
    "frustrated": ["repeated attempts", "sigh-like cues (visual)", "hands on head", "quick resets"],
    "neutral": ["steady posture", "calm movement", "no strong affect cues"],
}

def ensure_opencv():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is not installed. Install with: pip install opencv-python"
        )

def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_stem(path: str) -> str:
    return Path(path).stem.replace(" ", "_")

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def format_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m:02d}:{s:05.2f}"

def overlay_text(frame, lines: List[str], origin: Tuple[int, int] = (20, 40)) -> Any:
    """
    Draw simple overlay text on a frame. No fancy UI; judges just need clarity.
    """
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2

    # Background box (semi-opaque look via filled rectangle)
    line_height = 26
    box_w = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, scale, thickness)
        box_w = max(box_w, w)
    box_h = line_height * len(lines) + 18

    # Draw filled rect (dark)
    cv2.rectangle(frame, (x - 10, y - 30), (x + box_w + 20, y - 30 + box_h), (0, 0, 0), -1)

    # Put text (white)
    y_cursor = y
    for line in lines:
        cv2.putText(frame, line, (x, y_cursor), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_cursor += line_height

    return frame


# ----------------------------
# Cosmos calls
# ----------------------------

def call_cosmos_reason_stub(
    clip_frames: List[Any],
    requested_emotion: Optional[str] = None,
    seed: Optional[int] = None
) -> CosmosResult:
    """
    Offline stub that produces deterministic-ish outputs for demo videos.
    If requested_emotion is provided, it biases output toward it (useful if your
    input videos are already labeled angry/happy/etc).
    """
    rng = random.Random(seed if seed is not None else 12345)

    emotions = ["angry", "happy", "sad", "frustrated", "neutral"]
    if requested_emotion in emotions:
        emotion = requested_emotion
    else:
        # Mild bias based on filename label is handled outside; here choose random.
        emotion = rng.choice(emotions)

    confidence = clamp(rng.uniform(0.75, 0.95) + (0.03 if emotion != "neutral" else -0.05), 0.5, 0.99)
    evidence = rng.sample(EMOTION_EVIDENCE_BANK[emotion], k=min(3, len(EMOTION_EVIDENCE_BANK[emotion])))

    interaction_state = EMOTION_TO_STATE.get(emotion, "neutral")
    risk_level = "low"
    if emotion == "angry":
        risk_level = rng.choice(["medium", "high"])
    elif emotion == "frustrated":
        risk_level = rng.choice(["low", "medium"])
    elif emotion in ("sad",):
        risk_level = "low"
    elif emotion in ("happy",):
        risk_level = "low"

    actions = STATE_DEFAULT_ACTIONS.get(interaction_state, STATE_DEFAULT_ACTIONS["neutral"])
    return CosmosResult(
        emotion=emotion,
        confidence=confidence,
        evidence=evidence,
        interaction_state=interaction_state,
        risk_level=risk_level,
        recommended_robot_actions=actions,
    )

def call_cosmos_reason_api(
    clip_frames: List[Any],
    api_key: str,
    model: str = "cosmos-reason-2",
    extra_prompt: Optional[str] = None,
) -> CosmosResult:
    """
    TODO: Implement your real Cosmos Reason 2 call here.

    Because Cosmos endpoints/auth vary by environment, this is intentionally left
    as a placeholder. You can implement with:
    - NVIDIA hosted endpoint (if provided in the Cookoff docs/portal)
    - Your own service wrapper

    What you should do:
      1) Encode frames (e.g., JPEG) or an mp4 clip
      2) Send to the endpoint with a prompt that requests JSON output
      3) Parse the JSON response into CosmosResult
    """
    raise NotImplementedError(
        "API mode is not implemented. Use stub mode (default) or implement call_cosmos_reason_api()."
    )


# ----------------------------
# Video processing
# ----------------------------

def read_video_metadata(cap) -> Tuple[float, int, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return fps, total_frames, width, height

def get_frame_at(cap, frame_idx: int) -> Optional[Any]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None

def sample_clip_frames(
    cap,
    start_frame: int,
    end_frame: int,
    max_frames: int = 16
) -> List[Any]:
    """
    Uniformly sample up to max_frames from [start_frame, end_frame).
    """
    length = max(0, end_frame - start_frame)
    if length <= 0:
        return []

    if length <= max_frames:
        idxs = list(range(start_frame, end_frame))
    else:
        step = length / max_frames
        idxs = [start_frame + int(i * step) for i in range(max_frames)]

    frames: List[Any] = []
    for idx in idxs:
        frame = get_frame_at(cap, idx)
        if frame is not None:
            frames.append(frame)
    return frames

def infer_requested_emotion_from_filename(video_path: str) -> Optional[str]:
    name = Path(video_path).stem.lower()
    for key in ["angry", "happy", "sad", "frustrated", "neutral"]:
        if key in name:
            return key
    return None


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cosmos Cookoff demo runner (video -> reasoning -> overlay + JSON).")
    p.add_argument("--video", required=True, help="Path to input .mp4/.mov")
    p.add_argument("--outdir", default="demo/outputs", help="Output directory")
    p.add_argument("--window_s", type=float, default=3.0, help="Clip window length in seconds")
    p.add_argument("--stride_s", type=float, default=2.0, help="Stride between windows in seconds")
    p.add_argument("--max_windows", type=int, default=6, help="Max number of windows to process")
    p.add_argument("--overlay", action="store_true", help="Write an overlay .mp4 showing results")
    p.add_argument("--overlay_fps", type=float, default=30.0, help="FPS for overlay output (if enabled)")

    p.add_argument("--mode", choices=["stub", "api"], default="stub", help="stub=offline demo, api=call Cosmos endpoint")
    p.add_argument("--api_key_env", default="NVIDIA_API_KEY", help="Env var name holding Cosmos API key (api mode)")
    p.add_argument("--model", default="cosmos-reason-2", help="Model name (api mode)")
    p.add_argument("--seed", type=int, default=123, help="Random seed for stub mode outputs")

    return p.parse_args()

def main() -> int:
    args = parse_args()
    ensure_opencv()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    mkdirp(outdir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: unable to open video.", file=sys.stderr)
        return 2

    fps, total_frames, width, height = read_video_metadata(cap)
    if fps <= 0:
        fps = 30.0

    duration_s = (total_frames / fps) if total_frames > 0 else 0.0
    print(f"Loaded video: {video_path.name}")
    print(f"  fps={fps:.2f} frames={total_frames} duration={duration_s:.2f}s size={width}x{height}")

    requested_emotion = infer_requested_emotion_from_filename(str(video_path))

    # Overlay writer (optional)
    overlay_writer = None
    overlay_path = outdir / f"{safe_stem(str(video_path))}_overlay.mp4"
    if args.overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(str(overlay_path), fourcc, args.overlay_fps, (width, height))
        if not overlay_writer.isOpened():
            print("WARNING: could not open overlay writer. Disabling overlay output.", file=sys.stderr)
            overlay_writer = None

    # Determine windows
    window_frames = max(1, int(round(args.window_s * fps)))
    stride_frames = max(1, int(round(args.stride_s * fps)))

    results: List[Dict[str, Any]] = []
    processed = 0
    start_frame = 0

    while processed < args.max_windows:
        end_frame = min(total_frames, start_frame + window_frames) if total_frames else (start_frame + window_frames)
        if total_frames and start_frame >= total_frames:
            break

        clip_frames = sample_clip_frames(cap, start_frame, end_frame, max_frames=16)
        if not clip_frames:
            break

        t0_s = start_frame / fps
        t1_s = end_frame / fps

        # Call Cosmos (stub or api)
        if args.mode == "stub":
            seed = args.seed + processed
            cosmos_res = call_cosmos_reason_stub(
                clip_frames=clip_frames,
                requested_emotion=requested_emotion,
                seed=seed,
            )
        else:
            api_key = os.getenv(args.api_key_env, "")
            if not api_key:
                print(f"ERROR: API mode selected but env var {args.api_key_env} is not set.", file=sys.stderr)
                return 2
            cosmos_res = call_cosmos_reason_api(
                clip_frames=clip_frames,
                api_key=api_key,
                model=args.model,
            )

        rec = {
            "window": {
                "start_s": round(t0_s, 3),
                "end_s": round(t1_s, 3),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            },
            "cosmos": cosmos_res.to_json(),
        }
        results.append(rec)

        # Write per-window json
        win_json_path = outdir / f"{safe_stem(str(video_path))}_win{processed:02d}.json"
        with open(win_json_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print(f"Wrote: {win_json_path.name}  [{format_ts(t0_s)} - {format_ts(t1_s)}]  "
              f"emotion={cosmos_res.emotion} state={cosmos_res.interaction_state}")

        # Overlay: write frames from this window with the result printed
        if overlay_writer is not None and total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(start_frame, end_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                lines = [
                    f"Emotion: {cosmos_res.emotion}  (conf {cosmos_res.confidence:.2f})",
                    f"State: {cosmos_res.interaction_state}   Risk: {cosmos_res.risk_level}",
                    "Actions: " + ", ".join([a.action for a in cosmos_res.recommended_robot_actions[:3]]),
                ]
                frame = overlay_text(frame, lines)
                overlay_writer.write(frame)

        processed += 1
        start_frame += stride_frames

        # Stop early if we hit the end and have full metadata
        if total_frames and start_frame >= total_frames:
            break

    # Write combined json
    combined_path = outdir / f"{safe_stem(str(video_path))}_all.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video": str(video_path),
                "fps": round(float(fps), 3),
                "total_frames": int(total_frames),
                "requested_emotion_hint": requested_emotion,
                "windows": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote: {combined_path.name}")

    if overlay_writer is not None:
        overlay_writer.release()
        print(f"Wrote: {overlay_path.name}")

    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
