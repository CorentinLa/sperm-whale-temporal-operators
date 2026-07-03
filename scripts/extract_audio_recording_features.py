#!/usr/bin/env python3
"""
Extract click-timing features from DSWP audio recordings.

This script is intended to document and reproduce the acoustic feature table used in the
paper when a local copy of the Hugging Face dataset orrp/DSWP is available.
It accepts either files named *.wav or extensionless WAV containers.

Usage:
  python extract_audio_recording_features.py /path/to/DSWP_wavs --out data/dswp_audio_recording_features.csv

The detector is deliberately simple and transparent: it estimates a broadband energy
envelope, applies an adaptive threshold, detects local maxima with a refractory period,
and keeps clips with at least three detected clicks. The exact detector can be replaced
by a stronger coda detector; the analysis downstream uses only the resulting click times.
"""
from pathlib import Path
import argparse, wave, contextlib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture


def is_wav(path: Path) -> bool:
    try:
        with open(path, 'rb') as f:
            return f.read(4) == b'RIFF'
    except OSError:
        return False


def read_audio(path: Path):
    sr, x = wavfile.read(str(path))
    if x.dtype.kind in 'iu':
        x = x.astype(float) / max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
    else:
        x = x.astype(float)
    channels = 1 if x.ndim == 1 else x.shape[1]
    mono = x if x.ndim == 1 else x.mean(axis=1)
    return sr, channels, mono


def detect_clicks(x, sr):
    # Rectified energy envelope; smooth over 1 ms to stabilize impulsive peaks.
    env = np.abs(x)
    w = max(1, int(0.001 * sr))
    if w > 1:
        env = np.convolve(env, np.ones(w)/w, mode='same')
    # Adaptive threshold: robust floor plus upper percentile component.
    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    thr = max(med + 8*mad, np.percentile(env, 97.5))
    refractory = max(1, int(0.018 * sr))  # 18 ms: below the fastest coda ICI retained here.
    peaks, props = find_peaks(env, height=thr, distance=refractory)
    if len(peaks) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    # Keep up to 40 strongest peaks in temporal order; DSWP snippets are short codas.
    if len(peaks) > 40:
        order = np.argsort(props['peak_heights'])[-40:]
        peaks = np.sort(peaks[order])
    return peaks / sr, env[peaks]


def quantize_gap_symbols(gaps_s, centers_ms):
    centers = np.array(centers_ms) / 1000.0
    symbols = np.array(list('abcdef'))[:len(centers)]
    out = []
    for g in gaps_s:
        j = int(np.argmin(np.abs(np.log(g + 1e-12) - np.log(centers + 1e-12))))
        out.append(symbols[j])
    return ''.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio_dir', type=Path)
    ap.add_argument('--out', type=Path, default=Path('dswp_audio_recording_features.csv'))
    args = ap.parse_args()
    files = [p for p in args.audio_dir.rglob('*') if p.is_file() and is_wav(p)]
    preliminary = []
    gap_pool = []
    for p in files:
        try:
            sr, channels, x = read_audio(p)
        except Exception:
            continue
        t, amp = detect_clicks(x, sr)
        if len(t) < 3:
            continue
        gaps = np.diff(t)
        gap_pool.extend(gaps.tolist())
        preliminary.append((p.name, sr, channels, len(x)/sr, t, amp, gaps))
    if len(gap_pool) < 10:
        raise RuntimeError('Too few gaps detected.')
    logg = np.log(np.array(gap_pool)).reshape(-1, 1)
    gm = GaussianMixture(n_components=6, random_state=0, covariance_type='full').fit(logg)
    centers_s = np.sort(np.exp(gm.means_.ravel()))
    rows = []
    for name, sr, channels, duration, t, amp, gaps in preliminary:
        n = len(t)
        coda_dur = t[-1] - t[0]
        gs = quantize_gap_symbols(gaps, centers_s*1000)
        density_class = 'sparse_0_4' if n <= 4 else 'canonical_5_6' if n <= 6 else 'expanded_7_10' if n <= 10 else 'long_11_16' if n <= 16 else 'dense_17plus'
        rows.append({
            'name': name, 'sr': sr, 'channels': channels, 'duration_s': duration, 'nclick': n,
            'coda_dur_s': coda_dur, 'mean_ici_s': float(np.mean(gaps)), 'cv_ici': float(np.std(gaps)/np.mean(gaps)),
            'front2_mass': float(np.sum(gaps[:min(2, len(gaps))]) / np.sum(gaps)),
            'first_last_ratio': float(gaps[0]/gaps[-1]) if gaps[-1] > 0 else np.nan,
            'amp_cv': float(np.std(amp)/(np.mean(amp)+1e-12)),
            'amp_trend': float(np.polyfit(np.arange(len(amp)), amp, 1)[0]) if len(amp) > 1 else 0.0,
            'density_class': density_class,
            'gap_string': gs,
            'fast_share': float(sum(c in 'ab' for c in gs)/len(gs)) if gs else np.nan,
            'long_share': float(sum(c in 'ef' for c in gs)/len(gs)) if gs else np.nan,
        })
    df = pd.DataFrame(rows).sort_values('name')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'Wrote {len(df)} rows to {args.out}')
    print('Gap centers (ms):', ', '.join(f'{c*1000:.1f}' for c in centers_s))

if __name__ == '__main__':
    main()
