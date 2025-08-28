#!/usr/bin/env python3
"""
Generate spectrogram PNGs for GitHub Pages demo:
  audio/<id>_{clean,noisy,noise,ours,other}.wav|mp3|ogg|flac
→ spectrograms/<id>_{clean,noisy,noise,ours,other}_spec.png

All spectrograms share a *global* dB color scale, computed robustly
from the 1st/99th percentiles over all inputs, for fair visual comparison.
"""

import os, re, glob, argparse, math
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

# ----------------------------- CLI -----------------------------
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--audio_dir", default="audio", help="Input audio directory")
    p.add_argument("--out_dir",   default="spectrograms", help="Output image directory")
    p.add_argument("--suffixes",  default="clean,noisy,noise,ours,other",
                   help="Comma-separated list of expected suffixes per sample ID")
    p.add_argument("--sr",   type=int, default=16000, help="Target sample rate")
    p.add_argument("--n_fft",type=int, default=1024,  help="STFT n_fft")
    p.add_argument("--hop",  type=int, default=256,   help="STFT hop length")
    p.add_argument("--win",  default="hann",          help="Window type for STFT")
    p.add_argument("--cmap", default="magma",         help="Matplotlib colormap")
    p.add_argument("--title", action="store_true",    help="Draw a small title on each image")
    p.add_argument("--no_colorbar", action="store_true", help="Omit colorbar on images")
    p.add_argument("--axis", choices=["off","timehz","timelin"], default="off",
                   help="Axes style: off = no axes, timehz = time/Hz axes, timelin = time/linear frequency")
    p.add_argument("--db_low_high", default=None,
                   help="Override global dB range as 'low,high' (e.g., '-80,0'); if set, skip percentile pass")
    p.add_argument("--dpi", type=int, default=170, help="Image DPI")
    p.add_argument("--fig_w", type=float, default=4.0, help="Figure width (inches)")
    p.add_argument("--fig_h", type=float, default=2.2, help="Figure height (inches)")
    return p.parse_args()

# ------------------------ Helpers ------------------------------
def list_audio_files(audio_dir, suffixes):
    """Return dict: { (id, suffix) : path } and sorted list of unique ids."""
    exts = ("wav", "mp3", "ogg", "flac")
    pat  = re.compile(rf"^(.+)_({'|'.join(map(re.escape, suffixes))})\."
                      rf"({'|'.join(exts)})$", re.IGNORECASE)
    mapping = {}
    ids = set()
    for path in glob.glob(os.path.join(audio_dir, "*")):
        base = os.path.basename(path)
        m = pat.match(base)
        if not m:
            continue
        sid, suf, _ = m.groups()
        mapping[(sid, suf)] = path
        ids.add(sid)
    return mapping, sorted(ids)

def load_mono(path, sr):
    """Load audio in mono at target sr (librosa handles most formats)."""
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def compute_db(y, sr, n_fft, hop, win):
    """Return power spectrogram in dB."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win))**2
    S_db = librosa.power_to_db(S, ref=np.max)  # normalized to each file's max
    return S_db

def robust_db_bounds(S_db_list):
    """Robust global dB bounds from 1st/99th percentiles across files."""
    lows  = [np.percentile(S, 1)  for S in S_db_list]
    highs = [np.percentile(S, 99) for S in S_db_list]
    return float(min(lows)), float(max(highs))

def show_axes(ax, mode):
    if mode == "off":
        ax.set_axis_off()
        return dict(x_axis=None, y_axis=None)
    elif mode == "timehz":
        return dict(x_axis="time", y_axis="hz")
    elif mode == "timelin":
        return dict(x_axis="time", y_axis="linear")
    else:
        return dict(x_axis=None, y_axis=None)

# --------------------------- Main ------------------------------
def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    mapping, ids = list_audio_files(args.audio_dir, suffixes)

    if not ids:
        print(f"[!] No matching files found in {args.audio_dir}.")
        print(f"    Expected names like: <id>_{{{','.join(suffixes)}}}.wav")
        return

    print(f"Found {len(ids)} sample IDs:", ", ".join(ids))

    # --------- Pass 1: collect global dB bounds (unless overridden) ---------
    if args.db_low_high:
        vmin, vmax = map(float, args.db_low_high.split(","))
        print(f"Using user dB range: vmin={vmin:.1f}, vmax={vmax:.1f}")
    else:
        S_db_subset = []
        for sid in ids:
            for suf in suffixes:
                path = mapping.get((sid, suf))
                if not path: 
                    continue
                y, sr = load_mono(path, args.sr)
                if y.size == 0:
                    continue
                S_db = compute_db(y, sr, args.n_fft, args.hop, args.win)
                # sample a few columns to keep memory low
                if S_db.shape[1] > 400:
                    S_db = S_db[:, :: max(1, S_db.shape[1] // 400)]
                S_db_subset.append(S_db)
        if not S_db_subset:
            print("[!] Could not compute global dB bounds (no audio?). Aborting.")
            return
        vmin, vmax = robust_db_bounds(S_db_subset)
        print(f"Global dB bounds (robust): vmin={vmin:.1f}, vmax={vmax:.1f}")

    # --------- Pass 2: render spectrograms ---------
    for sid in ids:
        for suf in suffixes:
            path = mapping.get((sid, suf))
            if not path:
                continue
            y, sr = load_mono(path, args.sr)
            if y.size == 0:
                print(f"[!] Empty audio: {path}")
                continue

            S_db = compute_db(y, sr, args.n_fft, args.hop, args.win)

            fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
            ax  = plt.gca()
            axes_kw = show_axes(ax, args.axis)
            img = librosa.display.specshow(
                S_db, sr=sr, hop_length=args.hop,
                vmin=vmin, vmax=vmax, cmap=args.cmap,
                x_axis=axes_kw["x_axis"], y_axis=axes_kw["y_axis"], ax=ax
            )

            if args.title:
                ax.set_title(f"{sid} — {suf}", fontsize=9, pad=4)

            if not args.no_colorbar:
                cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Power (dB)", fontsize=8)

            plt.tight_layout()
            out_path = os.path.join(args.out_dir, f"{sid}_{suf}_spec.png")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
            plt.close(fig)
            print(f"[✓] {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()
