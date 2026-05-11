from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, filtfilt, welch


GOLD_STANDARD_NAME = "new_gold_standard"


@dataclass
class Config:
    fs: float = 100.0
    filter_low_hz: float = 2.5
    filter_high_hz: float = 12.5
    search_low_hz: float = 3.0
    search_high_hz: float = 10.0
    filter_order: int = 4
    welch_window_sec: float = 5.0
    welch_overlap_ratio: float = 0.5
    agreement_tolerance_hz: float = 0.5


@dataclass
class XsensSignal:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    packet_expected: int
    packet_actual: int
    packet_missing: int
    packet_duplicates: int
    packet_missing_ratio: float


@dataclass
class Spectrum:
    method: str
    peak_hz: float
    peak_power: float
    peak_snr: float
    dominance_ratio: float
    tremor_power: float
    band_power: float
    top_peaks: str
    freqs: np.ndarray
    psd: np.ndarray


def optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return float(text) if text else None


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    manifest_relative = base_dir / path
    if manifest_relative.exists():
        return manifest_relative
    return Path.cwd() / path


def packet_steps(counters: np.ndarray, modulus: int = 65536) -> np.ndarray:
    counters = np.asarray(counters, dtype=int)
    if len(counters) < 2:
        return np.asarray([], dtype=int)
    return ((counters[1:] - counters[:-1]) % modulus).astype(int)


def read_xsens(path: Path, config: Config, start_sec: float | None, end_sec: float | None) -> XsensSignal:
    if not path.exists():
        raise FileNotFoundError(f"Xsens file not found: {path}")

    df = pd.read_csv(path, sep="\t", comment="/")
    df.columns = [str(col).strip() for col in df.columns]
    required = ["PacketCounter", "Acc_X", "Acc_Y", "Acc_Z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid acceleration rows.")

    counters = df["PacketCounter"].to_numpy(dtype=int)
    steps = packet_steps(counters)
    duplicate_count = int(np.count_nonzero(steps == 0))
    keep_mask = np.r_[True, steps != 0]
    df = df.loc[keep_mask].reset_index(drop=True)

    counters = df["PacketCounter"].to_numpy(dtype=int)
    steps = packet_steps(counters)
    expanded_index = np.r_[0, np.cumsum(steps)].astype(int)
    packet_actual = int(len(df))
    packet_expected = int(expanded_index[-1] + 1) if len(expanded_index) else 0
    packet_missing = int(max(packet_expected - packet_actual, 0))
    packet_missing_ratio = packet_missing / packet_expected if packet_expected else 0.0

    full_index = np.arange(packet_expected)
    df = df.assign(_packet_index=expanded_index).set_index("_packet_index").reindex(full_index)
    df[["Acc_X", "Acc_Y", "Acc_Z"]] = df[["Acc_X", "Acc_Y", "Acc_Z"]].interpolate("linear").ffill().bfill()

    t = full_index / config.fs
    mask = np.ones(len(t), dtype=bool)
    if start_sec is not None:
        mask &= t >= start_sec
    if end_sec is not None:
        mask &= t <= end_sec
    if np.count_nonzero(mask) < max(50, int(2 * config.fs)):
        raise ValueError("Selected Xsens segment is too short.")

    return XsensSignal(
        t=t[mask].astype(float),
        x=df.loc[mask, "Acc_X"].to_numpy(dtype=float),
        y=df.loc[mask, "Acc_Y"].to_numpy(dtype=float),
        z=df.loc[mask, "Acc_Z"].to_numpy(dtype=float),
        packet_expected=packet_expected,
        packet_actual=packet_actual,
        packet_missing=packet_missing,
        packet_duplicates=duplicate_count,
        packet_missing_ratio=packet_missing_ratio,
    )


def bandpass(signal: np.ndarray, config: Config) -> np.ndarray:
    values = detrend(np.asarray(signal, dtype=float), type="linear")
    nyq = 0.5 * config.fs
    low = config.filter_low_hz / nyq
    high = min(config.filter_high_hz / nyq, 0.99)
    if not 0.0 < low < high:
        raise ValueError("Invalid band-pass settings.")
    b, a = butter(config.filter_order, [low, high], btype="bandpass")
    return filtfilt(b, a, values)


def first_pca_component(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[0]


def welch_psd(signals: list[np.ndarray], config: Config) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(len(signal) for signal in signals)
    nperseg = min(max(32, int(round(config.welch_window_sec * config.fs))), min_len)
    noverlap = min(int(round(nperseg * config.welch_overlap_ratio)), nperseg - 1)
    nfft = max(1024, 2 ** int(np.ceil(np.log2(nperseg * 4))))
    total_psd = None
    freqs = None
    for signal in signals:
        f, psd = welch(signal, fs=config.fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        freqs = f
        total_psd = psd if total_psd is None else total_psd + psd
    if freqs is None or total_psd is None:
        raise ValueError("No signals for PSD.")
    return freqs, total_psd


def local_peak_indices(values: np.ndarray) -> list[int]:
    peaks = []
    for idx in range(len(values)):
        left_ok = idx == 0 or values[idx] > values[idx - 1]
        right_ok = idx == len(values) - 1 or values[idx] >= values[idx + 1]
        if left_ok and right_ok:
            peaks.append(idx)
    return peaks or [int(np.argmax(values))]


def interpolate_peak(freqs: np.ndarray, psd: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(psd) - 1:
        return float(freqs[idx])
    eps = np.finfo(float).tiny
    alpha = np.log(max(float(psd[idx - 1]), eps))
    beta = np.log(max(float(psd[idx]), eps))
    gamma = np.log(max(float(psd[idx + 1]), eps))
    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-12:
        return float(freqs[idx])
    delta = float(np.clip(0.5 * (alpha - gamma) / denom, -1.0, 1.0))
    return float(freqs[idx] + delta * (freqs[1] - freqs[0]))


def integrate_power(psd: np.ndarray, freqs: np.ndarray) -> float:
    if len(psd) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd, freqs))
    return float(np.trapz(psd, freqs))


def summarize(method: str, freqs: np.ndarray, psd: np.ndarray, config: Config) -> Spectrum:
    band_mask = (freqs >= config.search_low_hz) & (freqs <= config.search_high_hz)
    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]
    if len(band_freqs) == 0:
        raise ValueError("No PSD bins in search band.")

    peaks = local_peak_indices(band_psd)
    ranked = sorted(peaks, key=lambda idx: float(band_psd[idx]), reverse=True)
    peak_idx = ranked[0]
    peak_hz = interpolate_peak(band_freqs, band_psd, peak_idx)
    peak_power = float(band_psd[peak_idx])
    noise_mask = np.abs(band_freqs - peak_hz) > 0.4
    noise_floor = float(np.median(band_psd[noise_mask])) if np.any(noise_mask) else float(np.median(band_psd))
    peak_snr = peak_power / noise_floor if noise_floor > 0 else float("inf")
    second_power = float(band_psd[ranked[1]]) if len(ranked) > 1 else 0.0
    dominance_ratio = peak_power / second_power if second_power > 0 else float("inf")
    tremor_mask = np.abs(freqs - peak_hz) <= 1.0
    top_peaks = ";".join(
        f"{rank}:{interpolate_peak(band_freqs, band_psd, idx):.3f}Hz,p={float(band_psd[idx]):.4g}"
        for rank, idx in enumerate(ranked[:5], start=1)
    )
    return Spectrum(
        method=method,
        peak_hz=peak_hz,
        peak_power=peak_power,
        peak_snr=peak_snr,
        dominance_ratio=dominance_ratio,
        tremor_power=integrate_power(psd[tremor_mask], freqs[tremor_mask]),
        band_power=integrate_power(band_psd, band_freqs),
        top_peaks=top_peaks,
        freqs=freqs,
        psd=psd,
    )


def analyze_gold(path: Path, config: Config, start_sec: float | None, end_sec: float | None) -> tuple[XsensSignal, dict[str, Spectrum]]:
    xsens = read_xsens(path, config, start_sec, end_sec)

    x_f = bandpass(xsens.x, config)
    y_f = bandpass(xsens.y, config)
    z_f = bandpass(xsens.z, config)

    svm = np.sqrt(xsens.x**2 + xsens.y**2 + xsens.z**2)
    svm = svm - np.median(svm)
    svm_f = bandpass(svm, config)

    pca_f = first_pca_component(np.column_stack([x_f, y_f, z_f]))

    spectra = {}
    for method, signals in {
        "svm": [svm_f],
        "pca_pc1": [pca_f],
    }.items():
        freqs, psd = welch_psd(signals, config)
        spectra[method] = summarize(method, freqs, psd, config)
    return xsens, spectra


def decide_reference(spectra: dict[str, Spectrum], config: Config) -> dict[str, object]:
    svm_hz = spectra["svm"].peak_hz
    pca_hz = spectra["pca_pc1"].peak_hz
    spread = abs(svm_hz - pca_hz)
    if spread <= config.agreement_tolerance_hz:
        return {
            "new_gold_peak_hz": float(np.median([svm_hz, pca_hz])),
            "new_gold_status": "accepted",
            "new_gold_reason": "svm_pca_agree",
            "new_gold_method_spread_hz": spread,
        }
    return {
        "new_gold_peak_hz": "",
        "new_gold_status": "rejected",
        "new_gold_reason": "svm_pca_disagree",
        "new_gold_method_spread_hz": spread,
    }


def row_from_result(
    trial_id: str,
    path: Path,
    xsens: XsensSignal,
    spectra: dict[str, Spectrum],
    config: Config,
    plot_path: Path,
) -> dict[str, object]:
    row = {
        "gold_standard_name": GOLD_STANDARD_NAME,
        "trial_id": trial_id,
        "gold_file": str(path),
        "fs": config.fs,
        "filter_low_hz": config.filter_low_hz,
        "filter_high_hz": config.filter_high_hz,
        "search_low_hz": config.search_low_hz,
        "search_high_hz": config.search_high_hz,
        "agreement_tolerance_hz": config.agreement_tolerance_hz,
        "duration_sec": float(xsens.t[-1] - xsens.t[0]) if len(xsens.t) > 1 else 0.0,
        "packet_expected": xsens.packet_expected,
        "packet_actual": xsens.packet_actual,
        "packet_missing": xsens.packet_missing,
        "packet_duplicates": xsens.packet_duplicates,
        "packet_missing_ratio": xsens.packet_missing_ratio,
        "plot": str(plot_path),
        **decide_reference(spectra, config),
    }
    for method, spectrum in spectra.items():
        row[f"{method}_peak_hz"] = spectrum.peak_hz
        row[f"{method}_peak_snr"] = spectrum.peak_snr
        row[f"{method}_dominance_ratio"] = spectrum.dominance_ratio
        row[f"{method}_tremor_power"] = spectrum.tremor_power
        row[f"{method}_band_power"] = spectrum.band_power
        row[f"{method}_top_peaks"] = spectrum.top_peaks
    return row


def save_plot(path: Path, trial_id: str, spectra: dict[str, Spectrum], config: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 5.2))
    colors = {"svm": "#1f77b4", "pca_pc1": "#d62728"}
    for method in ["svm", "pca_pc1"]:
        spectrum = spectra[method]
        psd = spectrum.psd / np.max(spectrum.psd) if np.max(spectrum.psd) > 0 else spectrum.psd
        plt.plot(spectrum.freqs, psd, label=f"{method}: {spectrum.peak_hz:.2f} Hz", color=colors[method], linewidth=2)
    plt.axvspan(config.search_low_hz, config.search_high_hz, color="gray", alpha=0.10, label="search band")
    plt.xlim(0, min(15.0, config.fs / 2.0))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized PSD")
    plt.title(f"New Gold Standard - {trial_id}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


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
        writer.writerows(rows)


def run_one(path: Path, trial_id: str, out_dir: Path, config: Config, start_sec: float | None, end_sec: float | None) -> dict[str, object]:
    xsens, spectra = analyze_gold(path, config, start_sec, end_sec)
    plot_path = out_dir / f"{trial_id}_new_gold_standard_psd.png"
    save_plot(plot_path, trial_id, spectra, config)
    row = row_from_result(trial_id, path, xsens, spectra, config, plot_path)
    json_path = out_dir / f"{trial_id}_new_gold_standard_result.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)
    row["result_json"] = str(json_path)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean new gold standard detector for Xsens IMU tremor frequency.")
    parser.add_argument("--gold", type=Path, help="One Xsens txt file.")
    parser.add_argument("--manifest", type=Path, help="CSV with gold column and optional trial_id/gold_start/gold_end.")
    parser.add_argument("--out-dir", type=Path, default=Path("results_new_gold_standard_clean"))
    parser.add_argument("--trial-id", type=str)
    parser.add_argument("--gold-start", type=float)
    parser.add_argument("--gold-end", type=float)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--filter-low", type=float, default=2.5)
    parser.add_argument("--filter-high", type=float, default=12.5)
    parser.add_argument("--search-low", type=float, default=3.0)
    parser.add_argument("--search-high", type=float, default=10.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--welch-window-sec", type=float, default=5.0)
    parser.add_argument("--welch-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--agreement-tolerance-hz", type=float, default=0.5)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        fs=args.fs,
        filter_low_hz=args.filter_low,
        filter_high_hz=args.filter_high,
        search_low_hz=args.search_low,
        search_high_hz=args.search_high,
        filter_order=args.filter_order,
        welch_window_sec=args.welch_window_sec,
        welch_overlap_ratio=args.welch_overlap_ratio,
        agreement_tolerance_hz=args.agreement_tolerance_hz,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        if "gold" not in manifest.columns:
            raise ValueError("Manifest must contain a gold column.")
        for index, item in manifest.iterrows():
            path = resolve_path(item["gold"], args.manifest.parent)
            trial_id = str(item.get("trial_id") or safe_stem(path) or f"row_{index + 1}")
            start_sec = optional_float(item.get("gold_start"))
            end_sec = optional_float(item.get("gold_end"))
            try:
                rows.append(run_one(path, trial_id, args.out_dir, config, start_sec, end_sec))
                print(f"{trial_id}: done")
            except Exception as exc:
                rows.append({"gold_standard_name": GOLD_STANDARD_NAME, "trial_id": trial_id, "gold_file": str(path), "error": str(exc)})
                print(f"{trial_id}: error: {exc}")
    else:
        if args.gold is None:
            raise ValueError("Use --gold or --manifest.")
        trial_id = args.trial_id or safe_stem(args.gold)
        rows.append(run_one(args.gold, trial_id, args.out_dir, config, args.gold_start, args.gold_end))

    summary_path = args.out_dir / "new_gold_standard_summary.csv"
    write_rows(summary_path, rows)
    print(f"Wrote {len(rows)} row(s) to {summary_path}")


if __name__ == "__main__":
    main()
