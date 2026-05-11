from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, welch


@dataclass
class VideoConfig:
    real_hand: str = "Right"
    mirror_video: bool = True
    lowcut_hz: float = 3.0
    highcut_hz: float = 12.0
    search_low_hz: float = 3.0
    search_high_hz: float = 10.0
    filter_order: int = 4
    welch_window_sec: float = 6.0
    welch_overlap_ratio: float = 0.5
    segment_parts: int = 4
    segment_tolerance_hz: float = 0.30
    min_valid_ratio: float = 0.35
    min_peak_prominence_std: float = 3.0
    harmonic_tolerance_hz: float = 0.35
    resize_height: int = 600
    video_start: float | None = None
    video_end: float | None = None
    anchor_feature: str = "tips_relative_wrist_pca"
    use_candidate_score_selection: bool = False


@dataclass
class VideoSignal:
    t: np.ndarray
    fs: float
    pca_signal: np.ndarray
    middle_signal: np.ndarray
    feature_signals: dict[str, np.ndarray]
    valid_mask: np.ndarray
    tracked_mask: np.ndarray
    valid_ratio: float
    tracking_ratio: float
    selected_mediapipe_hand: str
    coordinate_scale: float


@dataclass
class SpectrumResult:
    method: str
    peak_hz: float
    peak_power: float
    peak_snr: float
    peak_prominence_std: float
    band_power_ratio: float
    segment_stability_std_hz: float
    segment_stable_count: int
    reliable: bool
    top_peaks: str
    freqs: np.ndarray
    psd: np.ndarray


@dataclass
class PeakCandidate:
    feature: str
    peak_hz: float
    peak_power: float
    peak_snr: float
    peak_prominence_std: float
    segment_stability_std_hz: float
    segment_stable_count: int
    reliable: bool
    harmonic_source_hz: float | None = None
    harmonic_supported: bool = False


LANDMARK_IDS = {
    "wrist": 0,
    "index_mcp": 5,
    "middle_mcp": 9,
    "ring_mcp": 13,
    "pinky_mcp": 17,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_dip": 19,
    "pinky_tip": 20,
}

PALM_IDS = [0, 5, 9, 13, 17]


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return float(text) if text else None


def mapped_mediapipe_hand(real_hand: str, mirror_video: bool) -> str:
    if not mirror_video:
        return real_hand
    return "Left" if real_hand == "Right" else "Right"


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def interpolate_nan(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.interpolate("linear").ffill().bfill().to_numpy(dtype=float)


def pca_first_component(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[0]


def pca_components(matrix: np.ndarray, n_components: int = 3) -> list[np.ndarray]:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(n_components, vt.shape[0])
    return [centered @ vt[idx] for idx in range(count)]


def pca_after_bandpass(columns: list[np.ndarray], fs: float, config: VideoConfig) -> np.ndarray:
    filtered = [zero_phase_bandpass(column, fs, config) for column in columns]
    return pca_first_component(np.column_stack(filtered))


def in_time_range(t: float, start: float | None, end: float | None) -> bool:
    if start is not None and t < start:
        return False
    if end is not None and t > end:
        return False
    return True


def pick_target_hand(results: object, expected_label: str) -> object | None:
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None
    for idx, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        if label == expected_label:
            return results.multi_hand_landmarks[idx]
    return None


def extract_landmark_frame(landmarks: object) -> tuple[dict[str, tuple[float, float, float]], float] | None:
    points = landmarks.landmark
    palm_scale_candidates = [
        math.hypot(points[5].x - points[17].x, points[5].y - points[17].y),
        math.hypot(points[9].x - points[17].x, points[9].y - points[17].y),
        math.hypot(points[0].x - points[9].x, points[0].y - points[9].y),
        math.hypot(points[0].x - points[13].x, points[0].y - points[13].y),
    ]
    palm_scale_values = [value for value in palm_scale_candidates if np.isfinite(value) and value > 1e-6]
    if not palm_scale_values:
        return None
    palm_scale = float(np.median(palm_scale_values))

    raw_points = {}
    for name, idx in LANDMARK_IDS.items():
        lm = points[idx]
        raw_points[name] = (float(lm.x), float(lm.y), float(lm.z))
    return raw_points, palm_scale


def build_feature_matrix(frames: list[dict[str, tuple[float, float, float]] | None]) -> tuple[np.ndarray, np.ndarray]:
    columns = []
    for name in LANDMARK_IDS:
        columns.extend([f"{name}_x", f"{name}_y"])
    matrix = np.full((len(frames), len(columns)), np.nan, dtype=float)
    middle_y = np.full(len(frames), np.nan, dtype=float)

    for row_idx, frame in enumerate(frames):
        if frame is None:
            continue
        col_idx = 0
        for name in LANDMARK_IDS:
            x, y, _z = frame[name]
            matrix[row_idx, col_idx] = x
            matrix[row_idx, col_idx + 1] = y
            col_idx += 2
        middle_y[row_idx] = frame["middle_tip"][1]
    return matrix, middle_y


def build_named_feature_columns(
    frames: list[dict[str, tuple[float, float, float]] | None],
) -> dict[str, np.ndarray]:
    feature_names = [
        "wrist_x",
        "wrist_y",
        "wrist_z",
        "index_mcp_x",
        "index_mcp_y",
        "index_mcp_z",
        "middle_mcp_x",
        "middle_mcp_y",
        "middle_mcp_z",
        "ring_mcp_x",
        "ring_mcp_y",
        "ring_mcp_z",
        "pinky_mcp_x",
        "pinky_mcp_y",
        "pinky_mcp_z",
        "middle_tip_x",
        "middle_tip_y",
        "middle_tip_z",
        "ring_tip_x",
        "ring_tip_y",
        "ring_tip_z",
        "pinky_tip_x",
        "pinky_tip_y",
        "pinky_tip_z",
        "middle_dip_x",
        "middle_dip_y",
        "middle_dip_z",
        "ring_dip_x",
        "ring_dip_y",
        "ring_dip_z",
        "pinky_dip_x",
        "pinky_dip_y",
        "pinky_dip_z",
    ]
    features = {name: np.full(len(frames), np.nan, dtype=float) for name in feature_names}
    for row_idx, frame in enumerate(frames):
        if frame is None:
            continue
        for point_name in [
            "wrist",
            "index_mcp",
            "middle_mcp",
            "ring_mcp",
            "pinky_mcp",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
            "middle_dip",
            "ring_dip",
            "pinky_dip",
        ]:
            x, y, _z = frame[point_name]
            if f"{point_name}_x" in features:
                features[f"{point_name}_x"][row_idx] = x
            if f"{point_name}_y" in features:
                features[f"{point_name}_y"][row_idx] = y
            if f"{point_name}_z" in features:
                features[f"{point_name}_z"][row_idx] = _z
    return features


def reject_jump_frames(
    frames: list[dict[str, tuple[float, float, float]] | None],
    tracked_mask: np.ndarray,
) -> np.ndarray:
    valid = tracked_mask.copy()
    if len(frames) < 3:
        return valid

    xy_jumps = np.full(len(frames), np.nan, dtype=float)
    z_jumps = np.full(len(frames), np.nan, dtype=float)
    last_frame = None
    for idx, frame in enumerate(frames):
        if frame is None:
            continue
        if last_frame is not None:
            xy_diffs = []
            z_diffs = []
            for name in LANDMARK_IDS:
                x0, y0, z0 = last_frame[name]
                x1, y1, z1 = frame[name]
                xy_diffs.append(math.hypot(x1 - x0, y1 - y0))
                z_diffs.append(abs(z1 - z0))
            xy_jumps[idx] = float(np.median(xy_diffs))
            z_jumps[idx] = float(np.median(z_diffs))
        last_frame = frame

    xy_med = np.nanmedian(xy_jumps)
    z_med = np.nanmedian(z_jumps)
    xy_thr = max(0.35, float(xy_med + 8.0 * robust_mad(xy_jumps)))
    z_thr = max(0.55, float(z_med + 8.0 * robust_mad(z_jumps)))
    valid &= np.isnan(xy_jumps) | (xy_jumps <= xy_thr)
    valid &= np.isnan(z_jumps) | (z_jumps <= z_thr)
    return valid


def resample_uniform(times: np.ndarray, values: np.ndarray, target_fs: float) -> tuple[np.ndarray, np.ndarray]:
    start = float(times[0])
    end = float(times[-1])
    if end <= start:
        raise RuntimeError("Video timestamps are not usable.")
    uniform_t = np.arange(start, end, 1.0 / target_fs)
    if len(uniform_t) < 50:
        raise RuntimeError("Selected video segment is too short.")
    uniform_values = np.interp(uniform_t, times, values)
    return uniform_t - uniform_t[0], uniform_values


def extract_video_signal(video_path: Path, config: VideoConfig) -> VideoSignal:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    expected_label = mapped_mediapipe_hand(config.real_hand, config.mirror_video)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times: list[float] = []
    frames: list[dict[str, tuple[float, float, float]] | None] = []
    palm_scales: list[float] = []
    tracked: list[bool] = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if timestamp <= 0:
            timestamp = frame_idx / nominal_fps
        frame_idx += 1
        if not in_time_range(timestamp, config.video_start, config.video_end):
            continue

        h, w = frame.shape[:2]
        resized_w = int(w * (config.resize_height / h))
        frame_small = cv2.resize(frame, (resized_w, config.resize_height))
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        landmarks = pick_target_hand(results, expected_label)
        if landmarks is None:
            frames.append(None)
            palm_scales.append(np.nan)
            tracked.append(False)
        else:
            extracted = extract_landmark_frame(landmarks)
            if extracted is None:
                frames.append(None)
                palm_scales.append(np.nan)
                tracked.append(False)
            else:
                raw_points, palm_scale = extracted
                frames.append(raw_points)
                palm_scales.append(palm_scale)
                tracked.append(True)
        times.append(timestamp)

    cap.release()
    hands.close()

    if not times:
        raise RuntimeError("No video frames were selected.")

    times_np = np.asarray(times, dtype=float)
    tracked_mask = np.asarray(tracked, dtype=bool)
    valid_mask = reject_jump_frames(frames, tracked_mask)
    tracking_ratio = float(np.mean(tracked_mask))
    valid_ratio = float(np.mean(valid_mask))
    if valid_ratio < config.min_valid_ratio:
        raise RuntimeError(
            f"Valid hand-frame ratio is too low: {valid_ratio:.1%}. "
            f"real_hand={config.real_hand}, mediapipe={expected_label}"
        )

    palm_scales_np = np.asarray(palm_scales, dtype=float)
    scale_values = palm_scales_np[valid_mask & np.isfinite(palm_scales_np) & (palm_scales_np > 1e-6)]
    coordinate_scale = float(np.nanmedian(scale_values)) if len(scale_values) else 1.0

    matrix, middle_y = build_feature_matrix(frames)
    named_features = build_named_feature_columns(frames)
    matrix[~valid_mask, :] = np.nan
    middle_y[~valid_mask] = np.nan
    for values in named_features.values():
        values[~valid_mask] = np.nan
    matrix = matrix / coordinate_scale
    middle_y = middle_y / coordinate_scale
    for name in list(named_features):
        named_features[name] = named_features[name] / coordinate_scale
    middle_interp = interpolate_nan(middle_y)
    named_interp = {name: interpolate_nan(values) for name, values in named_features.items()}
    for point_name in ["middle_tip", "ring_tip", "pinky_tip"]:
        named_interp[f"{point_name}_rel_wrist_x"] = named_interp[f"{point_name}_x"] - named_interp["wrist_x"]
        named_interp[f"{point_name}_rel_wrist_y"] = named_interp[f"{point_name}_y"] - named_interp["wrist_y"]
        rel_xy = np.sqrt(
            named_interp[f"{point_name}_rel_wrist_x"] ** 2
            + named_interp[f"{point_name}_rel_wrist_y"] ** 2
        )
        rel_xyz = np.sqrt(
            named_interp[f"{point_name}_rel_wrist_x"] ** 2
            + named_interp[f"{point_name}_rel_wrist_y"] ** 2
            + (named_interp[f"{point_name}_z"] - named_interp["wrist_z"]) ** 2
        )
        named_interp[f"{point_name}_dist_wrist_xy"] = rel_xy
        named_interp[f"{point_name}_dist_wrist_xyz"] = rel_xyz

    median_dt = float(np.median(np.diff(times_np))) if len(times_np) > 2 else 1.0 / nominal_fps
    fs = 1.0 / median_dt if median_dt > 0 else nominal_fps
    fs = float(np.clip(fs, 15.0, 120.0))
    t_uniform, middle_uniform = resample_uniform(times_np, middle_interp, fs)
    feature_uniform = {}
    for name, values in named_interp.items():
        _, feature_uniform[name] = resample_uniform(times_np, values, fs)
    all_xy_columns = []
    for name in LANDMARK_IDS:
        all_xy_columns.extend([feature_uniform[f"{name}_x"], feature_uniform[f"{name}_y"]])
    tips_xy = []
    dips_xy = []
    wrist_tips_xy = []
    wrist_tips_xyz = []
    tips_relative_wrist_xy = []
    for point_name in ["middle_tip", "ring_tip", "pinky_tip"]:
        tips_xy.extend([feature_uniform[f"{point_name}_x"], feature_uniform[f"{point_name}_y"]])
        tips_relative_wrist_xy.extend(
            [
                feature_uniform[f"{point_name}_rel_wrist_x"],
                feature_uniform[f"{point_name}_rel_wrist_y"],
            ]
        )
    for point_name in ["middle_dip", "ring_dip", "pinky_dip"]:
        dips_xy.extend([feature_uniform[f"{point_name}_x"], feature_uniform[f"{point_name}_y"]])
    for point_name in ["wrist", "middle_tip", "ring_tip", "pinky_tip"]:
        wrist_tips_xy.extend([feature_uniform[f"{point_name}_x"], feature_uniform[f"{point_name}_y"]])
        wrist_tips_xyz.extend([feature_uniform[f"{point_name}_x"], feature_uniform[f"{point_name}_y"], feature_uniform[f"{point_name}_z"]])
    feature_uniform["tips_pca"] = pca_after_bandpass(tips_xy, fs, config)
    feature_uniform["dips_pca"] = pca_after_bandpass(dips_xy, fs, config)
    feature_uniform["wrist_tips_pca"] = pca_after_bandpass(wrist_tips_xy, fs, config)
    feature_uniform["wrist_tips_xyz_pca"] = pca_after_bandpass(wrist_tips_xyz, fs, config)
    feature_uniform["tips_relative_wrist_pca"] = pca_after_bandpass(tips_relative_wrist_xy, fs, config)
    rel_filtered = [zero_phase_bandpass(column, fs, config) for column in tips_relative_wrist_xy]
    for idx, component in enumerate(pca_components(np.column_stack(rel_filtered), n_components=3), start=1):
        feature_uniform[f"tips_relative_wrist_pca_pc{idx}"] = component
    feature_uniform["multi_landmark_pca"] = pca_after_bandpass(all_xy_columns, fs, config)
    pca_uniform = feature_uniform["multi_landmark_pca"]
    valid_uniform = np.interp(t_uniform + times_np[0], times_np, valid_mask.astype(float)) >= 0.5
    tracked_uniform = np.interp(t_uniform + times_np[0], times_np, tracked_mask.astype(float)) >= 0.5

    return VideoSignal(
        t=t_uniform,
        fs=fs,
        pca_signal=pca_uniform,
        middle_signal=middle_uniform,
        feature_signals=feature_uniform,
        valid_mask=valid_uniform,
        tracked_mask=tracked_uniform,
        valid_ratio=valid_ratio,
        tracking_ratio=tracking_ratio,
        selected_mediapipe_hand=expected_label,
        coordinate_scale=coordinate_scale,
    )


def zero_phase_bandpass(signal: np.ndarray, fs: float, config: VideoConfig) -> np.ndarray:
    signal = detrend(np.asarray(signal, dtype=float), type="linear")
    nyq = 0.5 * fs
    low = config.lowcut_hz / nyq
    high = min(config.highcut_hz / nyq, 0.99)
    if not 0 < low < high:
        raise ValueError("Invalid bandpass range for video sample rate.")
    b, a = butter(config.filter_order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def welch_psd(signal: np.ndarray, fs: float, config: VideoConfig) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(max(32, int(round(config.welch_window_sec * fs))), len(signal))
    noverlap = min(int(round(nperseg * config.welch_overlap_ratio)), nperseg - 1)
    nfft = max(1024, 2 ** int(np.ceil(np.log2(nperseg * 4))))
    return welch(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)


def local_peak_indices(values: np.ndarray) -> list[int]:
    indices = []
    for idx in range(len(values)):
        left_ok = idx == 0 or values[idx] > values[idx - 1]
        right_ok = idx == len(values) - 1 or values[idx] >= values[idx + 1]
        if left_ok and right_ok:
            indices.append(idx)
    return indices or [int(np.argmax(values))]


def interpolate_peak_frequency(freqs: np.ndarray, psd: np.ndarray, peak_idx: int) -> float:
    peak_hz = float(freqs[peak_idx])
    if peak_idx <= 0 or peak_idx >= len(psd) - 1:
        return peak_hz
    eps = np.finfo(float).tiny
    alpha = np.log(max(float(psd[peak_idx - 1]), eps))
    beta = np.log(max(float(psd[peak_idx]), eps))
    gamma = np.log(max(float(psd[peak_idx + 1]), eps))
    denominator = alpha - 2.0 * beta + gamma
    if abs(denominator) < 1e-12:
        return peak_hz
    delta = float(np.clip(0.5 * (alpha - gamma) / denominator, -1.0, 1.0))
    return peak_hz + delta * float(freqs[1] - freqs[0])


def integrate_power(psd: np.ndarray, freqs: np.ndarray) -> float:
    if len(psd) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd, freqs))
    return float(np.trapz(psd, freqs))


def required_stable_segments(config: VideoConfig) -> int:
    parts = max(3, int(config.segment_parts))
    return min(parts, max(3, math.ceil(parts * 0.75)))


def local_peak_prominence_std(
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    peak_idx: int,
    target_hz: float,
    config: VideoConfig,
) -> tuple[float, float, float]:
    local_radius_hz = max(0.8, config.segment_tolerance_hz * 3.0)
    exclude_radius_hz = max(0.20, config.segment_tolerance_hz * 0.75)
    local_mask = np.abs(band_freqs - target_hz) <= local_radius_hz
    baseline_mask = local_mask & (np.abs(band_freqs - target_hz) > exclude_radius_hz)
    if np.count_nonzero(baseline_mask) < 4:
        baseline_mask = np.abs(band_freqs - target_hz) > exclude_radius_hz
    baseline = band_psd[baseline_mask]
    if len(baseline) == 0:
        baseline = band_psd
    baseline_level = float(np.median(baseline))
    baseline_std = float(np.std(baseline))
    prominence = (float(band_psd[peak_idx]) - baseline_level) / baseline_std if baseline_std > 0 else float("inf")
    snr = float(band_psd[peak_idx]) / baseline_level if baseline_level > 0 else float("inf")
    return prominence, snr, baseline_level


def significant_peak_indices(
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    config: VideoConfig,
) -> list[int]:
    peaks = local_peak_indices(band_psd)
    significant = []
    for idx in peaks:
        peak_hz = interpolate_peak_frequency(band_freqs, band_psd, idx)
        prominence, _snr, _baseline = local_peak_prominence_std(band_freqs, band_psd, idx, peak_hz, config)
        if prominence >= config.min_peak_prominence_std:
            significant.append(idx)
    return significant


def has_significant_harmonic(
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    base_hz: float,
    config: VideoConfig,
) -> bool:
    harmonic_hz = base_hz * 2.0
    if harmonic_hz > config.search_high_hz:
        return False
    close = np.where(np.abs(band_freqs - harmonic_hz) <= config.harmonic_tolerance_hz)[0]
    if len(close) == 0:
        return False
    idx = int(close[np.argmax(band_psd[close])])
    prominence, _snr, _baseline = local_peak_prominence_std(band_freqs, band_psd, idx, float(band_freqs[idx]), config)
    return bool(prominence >= config.min_peak_prominence_std)


def resolve_harmonic_peak(
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    peak_idx: int,
    significant: list[int],
    config: VideoConfig,
) -> tuple[int, float | None, bool]:
    peak_hz = interpolate_peak_frequency(band_freqs, band_psd, peak_idx)
    half_hz = peak_hz / 2.0
    if half_hz < config.search_low_hz:
        return peak_idx, None, False
    half_candidates = [
        idx
        for idx in significant
        if abs(interpolate_peak_frequency(band_freqs, band_psd, idx) - half_hz) <= config.harmonic_tolerance_hz
    ]
    if not half_candidates:
        return peak_idx, None, False
    base_idx = max(half_candidates, key=lambda idx: float(band_psd[idx]))
    return base_idx, peak_hz, True


def segment_stability(clean_signal: np.ndarray, fs: float, target_hz: float, config: VideoConfig) -> tuple[float, int]:
    parts = max(3, int(config.segment_parts))
    if len(clean_signal) < parts * 50:
        return float("inf"), 0
    seg_len = len(clean_signal) // parts
    matched = []
    for part in range(parts):
        start = part * seg_len
        end = len(clean_signal) if part == parts - 1 else start + seg_len
        if end - start < 50:
            continue
        freqs, psd = welch_psd(clean_signal[start:end], fs, config)
        mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
        band_f = freqs[mask]
        band_p = psd[mask]
        if len(band_f) == 0:
            continue
        peaks = significant_peak_indices(band_f, band_p, config)
        if not peaks:
            continue
        ranked = sorted(peaks, key=lambda idx: float(band_p[idx]), reverse=True)
        best_idx, _harmonic_source_hz, _harmonic_supported = resolve_harmonic_peak(band_f, band_p, ranked[0], ranked, config)
        seg_peak_hz = interpolate_peak_frequency(band_f, band_p, best_idx)
        if abs(seg_peak_hz - target_hz) <= config.segment_tolerance_hz:
            matched.append(seg_peak_hz)
    if len(matched) < 2:
        return float("inf"), len(matched)
    return float(np.std(np.asarray(matched, dtype=float))), len(matched)


def harmonic_adjustment(
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    ranked: list[int],
    selected_idx: int,
    config: VideoConfig,
) -> int:
    selected_hz = float(band_freqs[selected_idx])
    half_hz = selected_hz / 2.0
    if half_hz < config.search_low_hz:
        return selected_idx
    for idx in ranked:
        hz = float(band_freqs[idx])
        if abs(hz - half_hz) <= config.harmonic_tolerance_hz and band_psd[idx] >= band_psd[selected_idx] * 0.30:
            return idx
    return selected_idx


def candidate_from_peak(
    feature: str,
    clean_signal: np.ndarray,
    fs: float,
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    peak_idx: int,
    config: VideoConfig,
    harmonic_source_hz: float | None = None,
    harmonic_supported: bool = False,
) -> PeakCandidate:
    peak_hz = interpolate_peak_frequency(band_freqs, band_psd, peak_idx)
    peak_power = float(band_psd[peak_idx])
    peak_prominence_std, peak_snr, _baseline = local_peak_prominence_std(band_freqs, band_psd, peak_idx, peak_hz, config)
    stability_std, stable_count = segment_stability(clean_signal, fs, peak_hz, config)
    reliable = bool(
        peak_prominence_std >= config.min_peak_prominence_std
        and stable_count >= required_stable_segments(config)
    )
    return PeakCandidate(
        feature=feature,
        peak_hz=peak_hz,
        peak_power=peak_power,
        peak_snr=peak_snr,
        peak_prominence_std=peak_prominence_std,
        segment_stability_std_hz=stability_std,
        segment_stable_count=stable_count,
        reliable=reliable,
        harmonic_source_hz=harmonic_source_hz,
        harmonic_supported=harmonic_supported,
    )


def candidate_from_band_psd(
    feature: str,
    band_freqs: np.ndarray,
    band_psd: np.ndarray,
    peak_idx: int,
    config: VideoConfig,
    segment_stable_count: int = 0,
    harmonic_source_hz: float | None = None,
    harmonic_supported: bool = False,
) -> PeakCandidate:
    peak_hz = interpolate_peak_frequency(band_freqs, band_psd, peak_idx)
    peak_power = float(band_psd[peak_idx])
    peak_prominence_std, peak_snr, _baseline = local_peak_prominence_std(band_freqs, band_psd, peak_idx, peak_hz, config)
    return PeakCandidate(
        feature=feature,
        peak_hz=peak_hz,
        peak_power=peak_power,
        peak_snr=peak_snr,
        peak_prominence_std=peak_prominence_std,
        segment_stability_std_hz=0.0,
        segment_stable_count=segment_stable_count,
        reliable=peak_prominence_std >= config.min_peak_prominence_std,
        harmonic_source_hz=harmonic_source_hz,
        harmonic_supported=harmonic_supported,
    )


def averaged_psd(signals: list[np.ndarray], fs: float, config: VideoConfig) -> tuple[np.ndarray, np.ndarray]:
    psds = []
    freqs_ref = None
    for signal in signals:
        clean = zero_phase_bandpass(signal, fs, config)
        freqs, psd = welch_psd(clean, fs, config)
        if freqs_ref is None:
            freqs_ref = freqs
        psds.append(psd)
    if freqs_ref is None or not psds:
        raise RuntimeError("No signals were provided for PSD averaging.")
    return freqs_ref, np.mean(np.vstack(psds), axis=0)


def psd_average_peak_candidates(
    feature: str,
    signals: list[np.ndarray],
    fs: float,
    config: VideoConfig,
) -> list[PeakCandidate]:
    freqs, psd = averaged_psd(signals, fs, config)
    mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
    band_f = freqs[mask]
    band_p = psd[mask]
    if len(band_f) == 0:
        return []
    significant = significant_peak_indices(band_f, band_p, config)
    ranked = sorted(significant, key=lambda idx: float(band_p[idx]), reverse=True)
    if not ranked:
        return []
    selected_idx, harmonic_source_hz, harmonic_supported = resolve_harmonic_peak(
        band_f,
        band_p,
        ranked[0],
        ranked,
        config,
    )
    return [
        candidate_from_band_psd(
            feature,
            band_f,
            band_p,
            selected_idx,
            config,
            harmonic_source_hz=harmonic_source_hz,
            harmonic_supported=harmonic_supported,
        )
    ]


def segment_psd_average_candidates(
    feature: str,
    signals: list[np.ndarray],
    fs: float,
    config: VideoConfig,
) -> list[PeakCandidate]:
    parts = max(3, int(config.segment_parts))
    if any(len(signal) < parts * 50 for signal in signals):
        return []
    seg_len = min(len(signal) for signal in signals) // parts
    candidates = []
    for part in range(parts):
        start = part * seg_len
        end = min(len(signal) for signal in signals) if part == parts - 1 else start + seg_len
        if end - start < 50:
            continue
        freqs, psd = averaged_psd([signal[start:end] for signal in signals], fs, config)
        mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
        band_f = freqs[mask]
        band_p = psd[mask]
        if len(band_f) == 0:
            continue
        significant = significant_peak_indices(band_f, band_p, config)
        ranked = sorted(significant, key=lambda idx: float(band_p[idx]), reverse=True)
        if not ranked:
            continue
        selected_idx, harmonic_source_hz, harmonic_supported = resolve_harmonic_peak(
            band_f,
            band_p,
            ranked[0],
            ranked,
            config,
        )
        candidates.append(
            candidate_from_band_psd(
                f"{feature}_seg{part + 1}",
                band_f,
                band_p,
                selected_idx,
                config,
                segment_stable_count=1,
                harmonic_source_hz=harmonic_source_hz,
                harmonic_supported=harmonic_supported,
            )
        )
    return candidates


def feature_peak_candidates(
    feature: str,
    raw_signal: np.ndarray,
    fs: float,
    config: VideoConfig,
    max_candidates: int = 5,
) -> list[PeakCandidate]:
    clean = zero_phase_bandpass(raw_signal, fs, config)
    freqs, psd = welch_psd(clean, fs, config)
    mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
    band_f = freqs[mask]
    band_p = psd[mask]
    if len(band_f) == 0:
        return []
    peaks = significant_peak_indices(band_f, band_p, config)
    ranked = sorted(peaks, key=lambda idx: float(band_p[idx]), reverse=True)[:max_candidates]
    if not ranked:
        return []
    selected_idx, harmonic_source_hz, harmonic_supported = resolve_harmonic_peak(
        band_f,
        band_p,
        ranked[0],
        ranked,
        config,
    )
    return [
        candidate_from_peak(
            feature,
            clean,
            fs,
            band_f,
            band_p,
            selected_idx,
            config,
            harmonic_source_hz=harmonic_source_hz,
            harmonic_supported=harmonic_supported,
        )
    ]


def multi_peak_candidates(
    feature: str,
    raw_signal: np.ndarray,
    fs: float,
    config: VideoConfig,
    max_candidates: int = 3,
) -> list[PeakCandidate]:
    clean = zero_phase_bandpass(raw_signal, fs, config)
    freqs, psd = welch_psd(clean, fs, config)
    mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
    band_f = freqs[mask]
    band_p = psd[mask]
    if len(band_f) == 0:
        return []
    peaks = significant_peak_indices(band_f, band_p, config) or local_peak_indices(band_p)
    ranked = sorted(peaks, key=lambda idx: float(band_p[idx]), reverse=True)[:max_candidates]
    candidates = []
    for idx in ranked:
        candidates.append(candidate_from_peak(feature, clean, fs, band_f, band_p, idx, config))
    return candidates


def segment_tip_peak_candidates(
    feature: str,
    raw_signal: np.ndarray,
    fs: float,
    config: VideoConfig,
) -> list[PeakCandidate]:
    clean = zero_phase_bandpass(raw_signal, fs, config)
    parts = max(3, int(config.segment_parts))
    if len(clean) < parts * 50:
        return []
    seg_len = len(clean) // parts
    candidates = []
    for part in range(parts):
        start = part * seg_len
        end = len(clean) if part == parts - 1 else start + seg_len
        segment = clean[start:end]
        if len(segment) < 50:
            continue
        freqs, psd = welch_psd(segment, fs, config)
        mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
        band_f = freqs[mask]
        band_p = psd[mask]
        if len(band_f) == 0:
            continue
        significant = significant_peak_indices(band_f, band_p, config)
        if not significant:
            continue
        ranked = sorted(significant, key=lambda idx: float(band_p[idx]), reverse=True)
        selected_idx, harmonic_source_hz, harmonic_supported = resolve_harmonic_peak(
            band_f,
            band_p,
            ranked[0],
            ranked,
            config,
        )
        peak_hz = interpolate_peak_frequency(band_f, band_p, selected_idx)
        prominence, peak_snr, _baseline = local_peak_prominence_std(band_f, band_p, selected_idx, peak_hz, config)
        candidates.append(
            PeakCandidate(
                feature=f"{feature}_seg{part + 1}",
                peak_hz=peak_hz,
                peak_power=float(band_p[selected_idx]),
                peak_snr=peak_snr,
                peak_prominence_std=prominence,
                segment_stability_std_hz=0.0,
                segment_stable_count=1,
                reliable=True,
                harmonic_source_hz=harmonic_source_hz,
                harmonic_supported=harmonic_supported,
            )
        )
    return candidates


def tip_name_from_vote(feature: str) -> str:
    if feature.startswith("tip_rel_wrist_psdavg"):
        return "tip_rel_wrist_psdavg"
    if feature.startswith("wrist_tips_pca"):
        return "wrist_tips_pca"
    if feature.startswith("tips_relative_wrist_pca"):
        return "tips_relative_wrist_pca"
    if feature.startswith("middle_tip_rel_wrist_y"):
        return "middle_tip_rel_wrist_y"
    if feature.startswith("ring_tip_rel_wrist_y"):
        return "ring_tip_rel_wrist_y"
    if feature.startswith("pinky_tip_rel_wrist_y"):
        return "pinky_tip_rel_wrist_y"
    if feature.startswith("middle_tip_y"):
        return "middle_tip_y"
    if feature.startswith("ring_tip_y"):
        return "ring_tip_y"
    if feature.startswith("pinky_tip_y"):
        return "pinky_tip_y"
    return feature


def segment_name_from_vote(feature: str) -> str:
    marker = "_seg"
    if marker not in feature:
        return ""
    return feature.rsplit(marker, 1)[-1]


def vote_peak_from_features(
    feature_signals: dict[str, np.ndarray],
    fs: float,
    config: VideoConfig,
    anchor_feature: str = "wrist_tips_pca",
) -> tuple[float | None, dict[str, object], list[PeakCandidate]]:
    validation_features = [
        anchor_feature,
        "middle_tip_y",
        "ring_tip_y",
        "pinky_tip_y",
    ]
    all_candidates: list[PeakCandidate] = []
    pca_full_candidates: list[PeakCandidate] = []
    if anchor_feature == "tip_rel_wrist_psdavg":
        rel_signals = [
            feature_signals["middle_tip_rel_wrist_y"],
            feature_signals["ring_tip_rel_wrist_y"],
            feature_signals["pinky_tip_rel_wrist_y"],
        ]
        pca_full_candidates = psd_average_peak_candidates(f"{anchor_feature}_full", rel_signals, fs, config)
        all_candidates.extend(pca_full_candidates)
        all_candidates.extend(segment_psd_average_candidates(anchor_feature, rel_signals, fs, config))
        validation_features = [
            "middle_tip_rel_wrist_y",
            "ring_tip_rel_wrist_y",
            "pinky_tip_rel_wrist_y",
        ]
    else:
        pca_signal = feature_signals.get(anchor_feature)
        if pca_signal is not None:
            pca_full_candidates = feature_peak_candidates(f"{anchor_feature}_full", pca_signal, fs, config)
            all_candidates.extend(pca_full_candidates)

    for feature in validation_features:
        signal = feature_signals.get(feature)
        if signal is None:
            continue
        all_candidates.extend(segment_tip_peak_candidates(feature, signal, fs, config))

    if not pca_full_candidates:
        return None, {
            "vote_status": f"no_{anchor_feature}_peak",
            "vote_feature_count": 0,
            "vote_support_features": "",
            "vote_peak_spread_hz": "",
        }, all_candidates

    anchor = pca_full_candidates[0]
    support = [
        candidate
        for candidate in all_candidates
        if candidate.feature != anchor.feature
        and abs(candidate.peak_hz - anchor.peak_hz) <= config.segment_tolerance_hz
    ]
    support_segments = {segment_name_from_vote(candidate.feature) for candidate in support}
    support_segments.discard("")
    medium_tolerance_hz = max(0.60, config.segment_tolerance_hz * 2.0)
    medium_support = [
        candidate
        for candidate in all_candidates
        if candidate.feature != anchor.feature
        and abs(candidate.peak_hz - anchor.peak_hz) <= medium_tolerance_hz
    ]
    medium_segments = {segment_name_from_vote(candidate.feature) for candidate in medium_support}
    medium_segments.discard("")
    if len(support) < 4:
        if len(medium_support) >= 4 and len(medium_segments) >= min(3, max(1, int(config.segment_parts))):
            freqs = np.asarray([candidate.peak_hz for candidate in medium_support], dtype=float)
            return anchor.peak_hz, {
                "vote_status": f"{anchor_feature}_medium_confidence",
                "vote_confidence": "medium",
                "vote_feature_count": len(medium_support),
                "vote_support_features": ";".join(sorted(candidate.feature for candidate in medium_support)),
                "vote_anchor_peak_hz": anchor.peak_hz,
                "vote_anchor_prominence_std": anchor.peak_prominence_std,
                "vote_distinct_source_count": len({tip_name_from_vote(candidate.feature) for candidate in medium_support}),
                "vote_distinct_segment_count": len(medium_segments),
                "vote_peak_spread_hz": float(np.max(freqs) - np.min(freqs)) if len(freqs) else "",
                "vote_median_prominence_std": float(np.median([candidate.peak_prominence_std for candidate in medium_support])),
                "vote_total_segment_support": len(medium_support),
                "vote_tolerance_hz": medium_tolerance_hz,
            }, all_candidates
        return None, {
            "vote_status": "insufficient_anchor_support",
            "vote_confidence": "reject",
            "vote_feature_count": len(support),
            "vote_support_features": ";".join(sorted(candidate.feature for candidate in support)),
            "vote_anchor_peak_hz": anchor.peak_hz,
            "vote_medium_feature_count": len(medium_support),
            "vote_medium_distinct_segment_count": len(medium_segments),
            "vote_peak_spread_hz": "",
        }, all_candidates
    if len(support_segments) < min(3, max(1, int(config.segment_parts))):
        return None, {
            "vote_status": "unstable_segment_support",
            "vote_confidence": "reject",
            "vote_feature_count": len(support),
            "vote_support_features": ";".join(sorted(candidate.feature for candidate in support)),
            "vote_anchor_peak_hz": anchor.peak_hz,
            "vote_distinct_segment_count": len(support_segments),
            "vote_medium_feature_count": len(medium_support),
            "vote_medium_distinct_segment_count": len(medium_segments),
            "vote_peak_spread_hz": "",
        }, all_candidates

    vote_sources = {candidate.feature for candidate in support}
    source_groups = {tip_name_from_vote(candidate.feature) for candidate in support}
    segments = support_segments
    freqs = np.asarray([candidate.peak_hz for candidate in support], dtype=float)
    spread = float(np.max(freqs) - np.min(freqs)) if len(freqs) else 0.0
    prominence = float(np.median([candidate.peak_prominence_std for candidate in support]))

    return anchor.peak_hz, {
        "vote_status": f"{anchor_feature}_validated",
        "vote_confidence": "high",
        "vote_feature_count": len(support),
        "vote_support_features": ";".join(sorted(vote_sources)),
        "vote_anchor_peak_hz": anchor.peak_hz,
        "vote_anchor_prominence_std": anchor.peak_prominence_std,
        "vote_distinct_source_count": len(source_groups),
        "vote_distinct_segment_count": len(segments),
        "vote_peak_spread_hz": spread,
        "vote_median_prominence_std": prominence,
        "vote_total_segment_support": len(support),
    }, all_candidates


def format_peak_candidates(candidates: list[PeakCandidate]) -> str:
    ordered = sorted(candidates, key=lambda item: (item.feature, -item.peak_power))
    return ";".join(
        f"{item.feature}:{item.peak_hz:.4g}Hz,prom={item.peak_prominence_std:.3g},seg={item.segment_stable_count},harm={int(item.harmonic_supported)},rel={int(item.reliable)}"
        for item in ordered
    )


def source_group(feature: str) -> str:
    for name in [
        "middle_tip",
        "ring_tip",
        "pinky_tip",
        "tips_relative_wrist_pca_pc1",
        "tips_relative_wrist_pca_pc2",
        "tips_relative_wrist_pca_pc3",
        "tips_relative_wrist_pca",
        "multi_landmark_pca",
    ]:
        if feature.startswith(name):
            return name
    return feature


def score_video_candidates(
    feature_signals: dict[str, np.ndarray],
    fs: float,
    config: VideoConfig,
) -> tuple[float | None, dict[str, object], list[PeakCandidate]]:
    candidate_feature_names = [
        "middle_tip_rel_wrist_y",
        "ring_tip_rel_wrist_y",
        "pinky_tip_rel_wrist_y",
        "middle_tip_dist_wrist_xy",
        "middle_tip_dist_wrist_xyz",
        "tips_relative_wrist_pca",
        "tips_relative_wrist_pca_pc1",
        "tips_relative_wrist_pca_pc2",
        "tips_relative_wrist_pca_pc3",
        "multi_landmark_pca",
    ]
    pollution_feature_names = ["wrist_x", "wrist_y", "middle_mcp_y"]

    candidates: list[PeakCandidate] = []
    for feature in candidate_feature_names:
        signal = feature_signals.get(feature)
        if signal is None:
            continue
        candidates.extend(multi_peak_candidates(feature, signal, fs, config, max_candidates=3))

    pollution_candidates: list[PeakCandidate] = []
    for feature in pollution_feature_names:
        signal = feature_signals.get(feature)
        if signal is None:
            continue
        pollution_candidates.extend(multi_peak_candidates(feature, signal, fs, config, max_candidates=2))

    if not candidates:
        return None, {
            "candidate_score_status": "no_candidates",
            "candidate_score_confidence": "reject",
        }, []

    tolerance = max(0.35, config.segment_tolerance_hz)
    remaining = sorted(candidates, key=lambda item: item.peak_power, reverse=True)
    clusters: list[list[PeakCandidate]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        keep = []
        for candidate in remaining:
            if abs(candidate.peak_hz - seed.peak_hz) <= tolerance:
                cluster.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        clusters.append(cluster)

    best_score = -1e9
    best_peak: float | None = None
    best_cluster: list[PeakCandidate] = []
    best_pollution: list[PeakCandidate] = []
    for cluster in clusters:
        freqs = np.asarray([candidate.peak_hz for candidate in cluster], dtype=float)
        peak_hz = float(np.median(freqs))
        groups = {source_group(candidate.feature) for candidate in cluster}
        finger_groups = {group for group in groups if group in {"middle_tip", "ring_tip", "pinky_tip"}}
        pca_groups = {group for group in groups if "pca" in group}
        distance_count = sum("dist_wrist" in candidate.feature for candidate in cluster)
        reliable_count = sum(candidate.reliable for candidate in cluster)
        median_prominence = float(np.median([candidate.peak_prominence_std for candidate in cluster]))
        pollution = [
            candidate
            for candidate in pollution_candidates
            if abs(candidate.peak_hz - peak_hz) <= tolerance and candidate.peak_prominence_std >= config.min_peak_prominence_std
        ]
        wrist_pollution = [candidate for candidate in pollution if candidate.feature.startswith("wrist_")]

        score = 0.0
        score += len(cluster) * 1.0
        score += len(groups) * 1.3
        score += len(finger_groups) * 1.0
        score += len(pca_groups) * 0.8
        score += distance_count * 0.5
        score += reliable_count * 0.8
        score += min(median_prominence, 12.0) * 0.15
        score -= len(wrist_pollution) * 2.5
        score -= max(0, len(pollution) - len(wrist_pollution)) * 1.2
        if len(groups) < 2:
            score -= 3.0
        if len(cluster) < 3:
            score -= 2.0

        if score > best_score:
            best_score = score
            best_peak = peak_hz
            best_cluster = cluster
            best_pollution = pollution

    if best_peak is None:
        return None, {
            "candidate_score_status": "no_scored_peak",
            "candidate_score_confidence": "reject",
        }, candidates

    best_groups = {source_group(candidate.feature) for candidate in best_cluster}
    confidence = "accepted" if best_score >= 8.0 and len(best_groups) >= 3 else "candidate"
    return best_peak, {
        "candidate_score_status": "scored_peak",
        "candidate_score_confidence": confidence,
        "candidate_score_peak_hz": best_peak,
        "candidate_score": best_score,
        "candidate_score_support_count": len(best_cluster),
        "candidate_score_source_count": len(best_groups),
        "candidate_score_support_features": ";".join(sorted(candidate.feature for candidate in best_cluster)),
        "candidate_score_pollution_features": ";".join(sorted(candidate.feature for candidate in best_pollution)),
    }, candidates + pollution_candidates


def assess_wrist_contamination(
    feature_signals: dict[str, np.ndarray],
    fs: float,
    config: VideoConfig,
    target_hz: float | None,
) -> dict[str, object]:
    if target_hz is None:
        return {
            "wrist_contamination_risk": "none",
            "wrist_contamination_support_count": 0,
            "wrist_contamination_support_features": "",
        }
    support = []
    for feature in ["wrist_x", "wrist_y", "middle_mcp_y"]:
        signal = feature_signals.get(feature)
        if signal is None:
            continue
        for candidate in multi_peak_candidates(feature, signal, fs, config, max_candidates=3):
            if (
                abs(candidate.peak_hz - target_hz) <= max(0.35, config.segment_tolerance_hz)
                and candidate.peak_prominence_std >= config.min_peak_prominence_std
            ):
                support.append(candidate)
                break
    support_features = {candidate.feature for candidate in support}
    wrist_xy_count = int("wrist_x" in support_features) + int("wrist_y" in support_features)
    if wrist_xy_count >= 2:
        risk = "high"
    elif wrist_xy_count == 1 and "middle_mcp_y" in support_features:
        risk = "medium"
    elif support:
        risk = "low"
    else:
        risk = "none"
    return {
        "wrist_contamination_risk": risk,
        "wrist_contamination_support_count": len(support),
        "wrist_contamination_support_features": ";".join(sorted(support_features)),
    }


def candidate_grade_from_score(score_summary: dict[str, object]) -> str:
    if score_summary.get("candidate_score_confidence") == "accepted":
        return "candidate"
    if score_summary.get("candidate_score_confidence") == "candidate":
        return "candidate"
    return "rejected"


def grade_video_result(
    final_peak_hz: float | None,
    final_reliable: bool,
    vote_summary: dict[str, object],
    wrist_risk_summary: dict[str, object],
    score_summary: dict[str, object],
) -> tuple[str, bool]:
    if final_peak_hz is None:
        return candidate_grade_from_score(score_summary), False
    if not final_reliable:
        return "candidate", False

    risk = str(wrist_risk_summary.get("wrist_contamination_risk", "none"))
    vote_confidence = str(vote_summary.get("vote_confidence", ""))
    vote_count = int(vote_summary.get("vote_feature_count") or 0)
    source_count = int(vote_summary.get("vote_distinct_source_count") or 0)
    segment_count = int(vote_summary.get("vote_distinct_segment_count") or 0)
    spread_value = vote_summary.get("vote_peak_spread_hz")
    try:
        spread = float(spread_value)
    except (TypeError, ValueError):
        spread = float("inf")

    strong_nonwrist_consensus = (
        vote_confidence == "high"
        and vote_count >= 10
        and source_count >= 4
        and segment_count >= 4
        and spread <= 0.45
    )
    weak_or_medium_consensus = vote_confidence != "high" or vote_count < 6 or source_count < 3

    if risk == "high" and not strong_nonwrist_consensus:
        return "candidate", False
    if risk == "medium" and weak_or_medium_consensus:
        return "candidate", False
    return "accepted", True


def analyze_signal(method: str, raw_signal: np.ndarray, fs: float, config: VideoConfig) -> SpectrumResult:
    clean = zero_phase_bandpass(raw_signal, fs, config)
    freqs, psd = welch_psd(clean, fs, config)
    mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
    band_f = freqs[mask]
    band_p = psd[mask]
    if len(band_f) == 0:
        raise RuntimeError("No PSD bins fall inside the tremor search band.")

    significant = significant_peak_indices(band_f, band_p, config)
    peaks = significant or local_peak_indices(band_p)
    ranked = sorted(peaks, key=lambda idx: float(band_p[idx]), reverse=True)
    selected_idx, _harmonic_source_hz, _harmonic_supported = resolve_harmonic_peak(band_f, band_p, ranked[0], ranked, config)
    peak_hz = interpolate_peak_frequency(band_f, band_p, selected_idx)
    peak_power = float(band_p[selected_idx])
    peak_prominence_std, peak_snr, _baseline = local_peak_prominence_std(band_f, band_p, selected_idx, peak_hz, config)
    total_power = integrate_power(psd, freqs)
    band_power = integrate_power(band_p, band_f)
    band_power_ratio = band_power / total_power if total_power > 0 else 0.0
    stability_std, stable_count = segment_stability(clean, fs, peak_hz, config)
    reliable = bool(
        peak_snr >= 3.0
        and peak_prominence_std >= config.min_peak_prominence_std
        and stable_count >= required_stable_segments(config)
    )
    top_peaks = ";".join(
        f"{rank}:{interpolate_peak_frequency(band_f, band_p, idx):.4g}Hz,p={float(band_p[idx]):.4g}"
        for rank, idx in enumerate(ranked[:5], start=1)
    )
    return SpectrumResult(
        method=method,
        peak_hz=peak_hz,
        peak_power=peak_power,
        peak_snr=peak_snr,
        peak_prominence_std=peak_prominence_std,
        band_power_ratio=band_power_ratio,
        segment_stability_std_hz=stability_std,
        segment_stable_count=stable_count,
        reliable=reliable,
        top_peaks=top_peaks,
        freqs=freqs,
        psd=psd,
    )


def choose_video_result(pca_result: SpectrumResult, middle_result: SpectrumResult) -> SpectrumResult:
    if pca_result.reliable:
        return pca_result
    if middle_result.reliable and middle_result.peak_snr > pca_result.peak_snr * 1.2:
        return middle_result
    return pca_result


def result_for_anchor_feature(
    anchor_feature: str,
    relative_wrist_pca_result: SpectrumResult,
    wrist_tips_pca_result: SpectrumResult,
    wrist_tips_xyz_pca_result: SpectrumResult,
) -> SpectrumResult:
    if anchor_feature == "tips_relative_wrist_pca":
        return relative_wrist_pca_result
    if anchor_feature == "wrist_tips_xyz_pca":
        return wrist_tips_xyz_pca_result
    return wrist_tips_pca_result


def save_plot(path: Path, trial_id: str, results: list[SpectrumResult], selected_method: str, config: VideoConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 5.2))
    for result in results:
        psd = result.psd / np.max(result.psd) if np.max(result.psd) > 0 else result.psd
        width = 2.2 if result.method == selected_method else 1.2
        alpha = 0.95 if result.method == selected_method else 0.65
        plt.plot(result.freqs, psd, linewidth=width, alpha=alpha, label=f"{result.method}: {result.peak_hz:.2f} Hz")
    plt.axvspan(config.search_low_hz, config.search_high_hz, color="gray", alpha=0.10, label="search band")
    plt.xlim(0, min(15.0, max(result.freqs[-1] for result in results)))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized PSD")
    plt.title(f"Clean Video Tremor Pipeline - {trial_id}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

    separate_dir = path.with_name(f"{path.stem}_separate_plots")
    separate_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        psd = result.psd / np.max(result.psd) if np.max(result.psd) > 0 else result.psd
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(result.freqs, psd, linewidth=2.0, color="#1f77b4")
        plt.axvspan(config.search_low_hz, config.search_high_hz, color="gray", alpha=0.10, label="search band")
        plt.axvline(result.peak_hz, color="#d62728", linestyle="--", linewidth=1.8, label=f"peak {result.peak_hz:.2f} Hz")
        plt.xlim(0, min(15.0, result.freqs[-1]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Normalized PSD")
        plt.title(f"{trial_id} - {result.method}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(separate_dir / f"{trial_id}_{result.method}_psd.png", dpi=220)
        plt.close()


def analyze_video(video_path: Path, trial_id: str, config: VideoConfig, out_dir: Path) -> dict[str, object]:
    video = extract_video_signal(video_path, config)
    pca_result = analyze_signal("multi_landmark_pca", video.pca_signal, video.fs, config)
    relative_wrist_pca_result = analyze_signal(
        "tips_relative_wrist_pca",
        video.feature_signals["tips_relative_wrist_pca"],
        video.fs,
        config,
    )
    wrist_tips_pca_result = analyze_signal("wrist_tips_pca", video.feature_signals["wrist_tips_pca"], video.fs, config)
    wrist_tips_xyz_pca_result = analyze_signal("wrist_tips_xyz_pca", video.feature_signals["wrist_tips_xyz_pca"], video.fs, config)
    middle_result = analyze_signal("middle_tip_y_fallback", video.middle_signal, video.fs, config)
    vote_peak, vote_summary, vote_candidates = vote_peak_from_features(
        video.feature_signals,
        video.fs,
        config,
        anchor_feature=config.anchor_feature,
    )
    xyz_vote_peak, xyz_vote_summary, xyz_vote_candidates = vote_peak_from_features(
        video.feature_signals,
        video.fs,
        config,
        anchor_feature="wrist_tips_xyz_pca",
    )
    scored_peak, score_summary, score_candidates = score_video_candidates(video.feature_signals, video.fs, config)
    fallback_selected = choose_video_result(pca_result, middle_result)
    anchor_result = result_for_anchor_feature(
        config.anchor_feature,
        relative_wrist_pca_result,
        wrist_tips_pca_result,
        wrist_tips_xyz_pca_result,
    )
    selected = fallback_selected
    final_method = selected.method
    final_peak_hz: float | None = selected.peak_hz
    final_reliable = False
    if vote_peak is not None:
        final_method = f"{config.anchor_feature}_validated"
        final_peak_hz = vote_peak
        final_reliable = True
        if (
            config.use_candidate_score_selection
            and
            scored_peak is not None
            and score_summary.get("candidate_score_confidence") == "accepted"
            and abs(float(scored_peak) - float(vote_peak)) > max(0.60, config.segment_tolerance_hz * 2.0)
        ):
            final_method = "candidate_score_accepted"
            final_peak_hz = scored_peak
            final_reliable = True
    else:
        if (
            config.use_candidate_score_selection
            and scored_peak is not None
            and score_summary.get("candidate_score_confidence") in {"accepted", "candidate"}
        ):
            final_method = f"candidate_score_{score_summary.get('candidate_score_confidence')}"
            final_peak_hz = scored_peak
            final_reliable = score_summary.get("candidate_score_confidence") == "accepted"
        else:
            final_method = "no_reliable_peak"
            final_peak_hz = None
    wrist_risk_summary = assess_wrist_contamination(video.feature_signals, video.fs, config, final_peak_hz)
    video_grade, final_reliable = grade_video_result(
        final_peak_hz,
        final_reliable,
        vote_summary,
        wrist_risk_summary,
        score_summary,
    )
    plot_path = out_dir / f"{trial_id}_clean_video_spectrum.png"
    save_plot(
        plot_path,
        trial_id,
        [relative_wrist_pca_result, wrist_tips_pca_result, wrist_tips_xyz_pca_result, pca_result, middle_result],
        final_method,
        config,
    )

    row: dict[str, object] = {
        "trial_id": trial_id,
        "video": str(video_path),
        "real_hand": config.real_hand,
        "mirror_video": config.mirror_video,
        "mediapipe_hand": video.selected_mediapipe_hand,
        "video_fs": video.fs,
        "tracking_ratio": video.tracking_ratio,
        "valid_ratio_after_jump_filter": video.valid_ratio,
        "coordinate_scale": video.coordinate_scale,
        "selected_method": final_method,
        "video_peak_hz": final_peak_hz,
        "video_grade": video_grade,
        "fallback_selected_method": fallback_selected.method,
        "fallback_peak_hz": fallback_selected.peak_hz,
        "video_reliable": final_reliable,
        "video_peak_snr": anchor_result.peak_snr if final_reliable else selected.peak_snr,
        "video_peak_prominence_std": anchor_result.peak_prominence_std if final_reliable else selected.peak_prominence_std,
        "video_segment_stability_std_hz": anchor_result.segment_stability_std_hz if final_reliable else selected.segment_stability_std_hz,
        "video_segment_stable_count": anchor_result.segment_stable_count if final_reliable else selected.segment_stable_count,
        **vote_summary,
        "feature_peak_candidates": format_peak_candidates(vote_candidates),
        **wrist_risk_summary,
        **score_summary,
        "alternative_candidate_peak_hz": scored_peak,
        "alternative_candidate_grade": candidate_grade_from_score(score_summary),
        "candidate_score_all_candidates": format_peak_candidates(score_candidates),
        "tips_relative_wrist_pca_peak_hz": relative_wrist_pca_result.peak_hz,
        "tips_relative_wrist_pca_reliable": relative_wrist_pca_result.reliable,
        "tips_relative_wrist_pca_top_peaks": relative_wrist_pca_result.top_peaks,
        "wrist_tips_pca_peak_hz": wrist_tips_pca_result.peak_hz,
        "wrist_tips_pca_reliable": wrist_tips_pca_result.reliable,
        "wrist_tips_pca_top_peaks": wrist_tips_pca_result.top_peaks,
        "wrist_tips_xyz_pca_peak_hz": wrist_tips_xyz_pca_result.peak_hz,
        "wrist_tips_xyz_pca_reliable": wrist_tips_xyz_pca_result.reliable,
        "wrist_tips_xyz_pca_top_peaks": wrist_tips_xyz_pca_result.top_peaks,
        "xyz_vote_peak_hz": xyz_vote_peak,
        "xyz_vote_status": xyz_vote_summary.get("vote_status", ""),
        "xyz_vote_feature_count": xyz_vote_summary.get("vote_feature_count", ""),
        "xyz_vote_support_features": xyz_vote_summary.get("vote_support_features", ""),
        "xyz_vote_distinct_segment_count": xyz_vote_summary.get("vote_distinct_segment_count", ""),
        "xyz_feature_peak_candidates": format_peak_candidates(xyz_vote_candidates),
        "pca_peak_hz": pca_result.peak_hz,
        "pca_reliable": pca_result.reliable,
        "pca_top_peaks": pca_result.top_peaks,
        "middle_peak_hz": middle_result.peak_hz,
        "middle_reliable": middle_result.reliable,
        "middle_top_peaks": middle_result.top_peaks,
        "plot": str(plot_path),
    }
    json_path = out_dir / f"{trial_id}_clean_video_result.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)
    row["result_json"] = str(json_path)
    return row


def paired_gold_path_for_real_hand(gold_path: Path, real_hand: str) -> Path:
    sensor_id = "00B47B1C" if real_hand == "Right" else "00B47B15"
    name = gold_path.name
    if "00B47B1C" in name:
        return gold_path.with_name(name.replace("00B47B1C", sensor_id))
    if "00B47B15" in name:
        return gold_path.with_name(name.replace("00B47B15", sensor_id))
    return gold_path


def resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        manifest_relative = base_dir / path
        cwd_relative = Path.cwd() / path
        path = manifest_relative if manifest_relative.exists() else cwd_relative
    return path


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def config_from_args(args: argparse.Namespace, real_hand: str | None = None) -> VideoConfig:
    return VideoConfig(
        real_hand=real_hand or args.real_hand,
        mirror_video=args.mirror_video,
        lowcut_hz=args.lowcut,
        highcut_hz=args.highcut,
        search_low_hz=args.search_low,
        search_high_hz=args.search_high,
        filter_order=args.filter_order,
        welch_window_sec=args.welch_window_sec,
        welch_overlap_ratio=args.welch_overlap_ratio,
        segment_parts=args.segment_parts,
        segment_tolerance_hz=args.segment_tolerance_hz,
        min_valid_ratio=args.min_valid_ratio,
        min_peak_prominence_std=args.min_peak_prominence_std,
        harmonic_tolerance_hz=args.harmonic_tolerance_hz,
        resize_height=args.resize_height,
        video_start=args.video_start,
        video_end=args.video_end,
        anchor_feature=args.anchor_feature,
        use_candidate_score_selection=args.use_candidate_score_selection,
    )


def run_manifest(args: argparse.Namespace) -> list[dict[str, object]]:
    manifest = pd.read_csv(args.manifest)
    if "video" not in manifest.columns:
        raise ValueError("Manifest must contain a video column.")
    base_dir = args.manifest.parent
    rows = []
    for row_idx, row in manifest.iterrows():
        video_path = resolve_path(row["video"], base_dir)
        trial_id = str(row.get("trial_id") or f"row_{row_idx + 1}")
        mirror_video = bool(row.get("mirror_video", args.mirror_video))
        hands_to_run = ["Right", "Left"] if args.both_real_hands else [str(row.get("real_hand") or args.real_hand)]
        for real_hand in hands_to_run:
            config = config_from_args(args, real_hand=real_hand)
            config.mirror_video = mirror_video
            suffix = "real_right_1c" if real_hand == "Right" else "real_left_15"
            run_id = f"{trial_id}_{suffix}" if args.both_real_hands else trial_id
            try:
                result = analyze_video(video_path, run_id, config, args.out_dir)
                if "gold" in row and not pd.isna(row["gold"]):
                    gold_path = paired_gold_path_for_real_hand(resolve_path(row["gold"], base_dir), real_hand)
                    result["gold"] = str(gold_path)
                    result["expected_sensor"] = "00B47B1C" if real_hand == "Right" else "00B47B15"
                rows.append(result)
                if result["video_peak_hz"] is None:
                    print(f"{run_id}: no reliable peak, {result['selected_method']}")
                else:
                    print(f"{run_id}: {result['video_peak_hz']:.2f} Hz, {result['selected_method']}")
            except Exception as exc:
                rows.append(
                    {
                        "trial_id": run_id,
                        "video": str(video_path),
                        "real_hand": real_hand,
                        "mirror_video": mirror_video,
                        "error": str(exc),
                    }
                )
                print(f"{run_id}: failed: {exc}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean video-only tremor frequency pipeline.")
    parser.add_argument("--video", type=Path, help="Path to one video file.")
    parser.add_argument("--manifest", type=Path, help="CSV containing video and optional trial_id/real_hand/mirror_video/gold.")
    parser.add_argument("--out-dir", type=Path, default=Path("results_clean_video_pipeline"))
    parser.add_argument("--trial-id", type=str, default=None)
    parser.add_argument("--real-hand", choices=["Right", "Left"], default="Right")
    parser.add_argument("--mirror-video", action="store_true", default=True)
    parser.add_argument("--not-mirrored", action="store_false", dest="mirror_video")
    parser.add_argument("--both-real-hands", action="store_true")
    parser.add_argument("--video-start", type=float, default=None)
    parser.add_argument("--video-end", type=float, default=None)
    parser.add_argument("--lowcut", type=float, default=3.0)
    parser.add_argument("--highcut", type=float, default=12.0)
    parser.add_argument("--search-low", type=float, default=3.0)
    parser.add_argument("--search-high", type=float, default=10.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--welch-window-sec", type=float, default=6.0)
    parser.add_argument("--welch-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--segment-parts", type=int, default=4)
    parser.add_argument("--segment-tolerance-hz", type=float, default=0.30)
    parser.add_argument("--min-valid-ratio", type=float, default=0.35)
    parser.add_argument("--min-peak-prominence-std", type=float, default=3.0)
    parser.add_argument("--harmonic-tolerance-hz", type=float, default=0.35)
    parser.add_argument("--resize-height", type=int, default=600)
    parser.add_argument(
        "--anchor-feature",
        choices=["wrist_tips_pca", "tips_relative_wrist_pca", "wrist_tips_xyz_pca", "tip_rel_wrist_psdavg"],
        default="tips_relative_wrist_pca",
    )
    parser.add_argument(
        "--use-candidate-score-selection",
        action="store_true",
        help="Experimental: allow candidate-score logic to override the original final peak selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        rows = run_manifest(args)
    else:
        if args.video is None:
            raise ValueError("Use --video or --manifest.")
        config = config_from_args(args)
        trial_id = args.trial_id or f"{safe_stem(args.video)}_{args.real_hand.lower()}"
        rows = [analyze_video(args.video, trial_id, config, args.out_dir)]
        if rows[0]["video_peak_hz"] is None:
            print(f"{trial_id}: no reliable peak, {rows[0]['selected_method']}")
        else:
            print(f"{trial_id}: {rows[0]['video_peak_hz']:.2f} Hz, {rows[0]['selected_method']}")
    write_rows(args.out_dir / "clean_video_summary.csv", rows)
    print(f"Wrote {len(rows)} row(s) to {args.out_dir / 'clean_video_summary.csv'}")


if __name__ == "__main__":
    main()
