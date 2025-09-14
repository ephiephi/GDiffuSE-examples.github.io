#!/usr/bin/env python3
"""
Compute PESQ, STOI, and SI-SDR for speech enhancement samples.

Directory layout (default: ./audio):
  audio/<ID>_clean.wav
  audio/<ID>_noisy.wav
  audio/<ID>_ours.wav
  audio/<ID>_other.wav
(extensions .wav/.flac/.ogg/.mp3 are supported; WAV recommended.)

Outputs:
  metrics.json  # { "<ID>": { "noisy": {...}, "ours": {...}, "other": {...} }, ... }
  metrics.csv   # sample_id,type,pesq,stoi,sisdr
"""

import os, re, glob, json, math, argparse
import numpy as np
import librosa
from pesq import pesq            # pip install pesq
from pystoi import stoi          # pip install pystoi

# ---------------------- Config ----------------------
EVAL_TYPES = ["noisy", "ours", "other"]   # which estimates to evaluate
AUDIO_EXTS = ("wav", "flac", "ogg", "mp3")

# ---------------------- Audio I/O -------------------
def find_file(audio_dir, sample_id, suffix):
    """Return first matching file path for <id>_<suffix>.<ext> or None."""
    for ext in AUDIO_EXTS:
        cand = os.path.join(audio_dir, f"{sample_id}_{suffix}.{ext}")
        if os.path.isfile(cand):
            return cand
    return None

def load_mono(path, sr):
    """Load audio as mono at target sample rate."""
    # librosa.load handles most formats; converts to float32 in [-1,1]
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

# ---------------------- Metrics ---------------------
def si_sdr(x, s, eps=1e-8):
    """
    Scale-Invariant SDR (in dB), where:
      s = reference (clean), x = estimate (enhanced)
    """
    x = x - np.mean(x)
    s = s - np.mean(s)
    # project x onto s
    alpha = np.sum(x * s) / (np.sum(s * s) + eps)
    s_target = alpha * s
    e_noise = x - s_target
    return 10.0 * np.log10((np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps))

def safe_pesq(fs, ref, deg):
    """
    PESQ wrapper: returns float or None (if clip too short or PESQ errors).
    Use wideband ('wb') for 16 kHz; narrowband ('nb') for 8 kHz.
    """
    mode = "wb" if fs >= 16000 else "nb"
    try:
        return float(pesq(fs, ref, deg, mode))
    except Exception as e:
        # PESQ fails for very short clips or non-speech-like inputs
        return None

def safe_stoi(fs, ref, deg):
    """
    STOI wrapper: returns float or None. Use extended STOI for fs up to 48 kHz.
    """
    try:
        return float(stoi(ref, deg, fs, extended=True))
    except Exception:
        return None

# ---------------------- Scan IDs --------------------
def discover_ids(audio_dir):
    """
    Discover sample IDs by scanning for *_clean.<ext> files.
    """
    ids = set()
    for ext in AUDIO_EXTS:
        for p in glob.glob(os.path.join(audio_dir, f"*_{'clean'}.{ext}")):
            base = os.path.basename(p)
            sid = re.sub(r"_clean\.[^.]+$", "", base)
            ids.add(sid)
    return sorted(ids)

# ---------------------- Main ------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute PESQ/STOI/SI-SDR metrics.")
    ap.add_argument("--audio_dir", default="audio", help="Directory with audio files")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (16k recommended)")
    ap.add_argument("--out_json", default="metrics.json", help="Output JSON path")
    ap.add_argument("--out_csv",  default="metrics.csv",  help="Output CSV path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv)  or ".", exist_ok=True)

    sample_ids = discover_ids(args.audio_dir)
    if not sample_ids:
        print(f"[!] No *_clean files found in {args.audio_dir}")
        return

    all_metrics = {}
    rows = []  # for CSV

    print(f"Found {len(sample_ids)} sample IDs:", ", ".join(sample_ids))

    for sid in sample_ids:
        clean_path = find_file(args.audio_dir, sid, "clean")
        if not clean_path:
            print(f"[!] Missing clean file for {sid}; skipping this ID.")
            continue

        ref, fs = load_mono(clean_path, args.sr)
        if ref.size < fs * 0.25:
            print(f"[!] {sid}: too short (<0.25s); metrics may be None.")

        all_metrics[sid] = {}

        for t in EVAL_TYPES:
            est_path = find_file(args.audio_dir, sid, t)
            if not est_path:
                print(f"[i] {sid}: missing '{t}' file; skipping this type.")
                continue

            deg, _ = load_mono(est_path, args.sr)

            # trim/pad to same length for fair metrics (recommended)
            L = min(ref.size, deg.size)
            if L == 0:
                print(f"[!] {sid}/{t}: empty audio; skipping.")
                continue
            r = ref[:L]
            d = deg[:L]

            m_pesq  = safe_pesq(fs, r, d)
            m_stoi  = safe_stoi(fs, r, d)
            m_sisdr = si_sdr(d, r)

            all_metrics[sid][t] = {
                "pesq":  None if m_pesq  is None else round(m_pesq,  4),
                "stoi":  None if m_stoi  is None else round(m_stoi,  4),
                "sisdr": None if m_sisdr is None else round(m_sisdr, 4),
            }

            rows.append([sid, t,
                         "" if m_pesq  is None else f"{m_pesq:.4f}",
                         "" if m_stoi  is None else f"{m_stoi:.4f}",
                         "" if m_sisdr is None else f"{m_sisdr:.4f}"])

    # ---------- write JSON in the exact shape your site expects ----------
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"[✓] Wrote {args.out_json}")

    # ---------- also write a CSV ----------
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "type", "pesq", "stoi", "sisdr"])
        w.writerows(rows)
    print(f"[✓] Wrote {args.out_csv}")

    # ---------- print a paste-ready JS snippet (optional) ----------
    print("\n// Paste into your page if you prefer inline JS instead of fetch('metrics.json'):\nMETRICS = ")
    print(json.dumps(all_metrics, indent=2))
    print(";")

if __name__ == "__main__":
    main()
