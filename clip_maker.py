#!/usr/bin/env python3
"""Auto-generate short vertical clips from a full episode's audio + SRT."""

import argparse
import subprocess
import re
from pathlib import Path
import pysubs2


KEYWORDS = {
    r"\bbeta\b": 3.0,
    r"\bguardrail[s]?\b": 3.0,
    r"\bignore(d)? instruction[s]?\b": 2.5,
    r"\b(spicy mode|explicit)\b": 2.5,
    r"\bmisuse\b": 2.2,
    r"\b(reliab(le|ility)|robust)\b": 2.0,
    r"\biterate|iteration\b": 1.6,
    r"\b(prompt engineering)\b": 1.6,
    r"\b(stumble|fail|artifact)\b": 1.6,
    r"\bpublic release|rollout\b": 1.6,
    r"\btable\b|\bzoom out\b|\bgroup size\b": 1.4,
    r"\bcreative superpower\b": 1.2,
    r"\bresponsible\b|\bethic(al|s)\b": 1.2,
}


def score_line(text: str) -> float:
    s = 0.0
    low = text.lower()
    for pat, w in KEYWORDS.items():
        if re.search(pat, low):
            s += w
    n = len(low.split())
    if 6 <= n <= 20: s += 1.2
    elif n < 4: s -= 0.8
    elif n > 36: s -= 0.6
    return s


def merge_window(events, idx, window_ms):
    start = events[idx].start
    end = events[idx].end
    i = idx - 1
    while i >= 0 and (start - events[i].end) <= window_ms:
        start = min(start, events[i].start)
        i -= 1
    j = idx + 1
    while j < len(events) and (events[j].start - end) <= window_ms:
        end = max(end, events[j].end)
        j += 1
    return start, end


def format_ts(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_clip(in_audio: Path, in_srt: Path, out_mp4: Path, start_ms: int, end_ms: int, w=1080, h=1920,
                font="Arial", bg="black", wf_ratio=0.35):
    # Trim audio
    out_aac = out_mp4.with_suffix(".aac")
    ss = format_ts(start_ms)
    to = format_ts(end_ms - start_ms)
    subprocess.check_call([
        "ffmpeg","-y","-ss", ss, "-t", to, "-i", str(in_audio),
        "-c:a","aac","-b:a","192k", str(out_aac)
    ])
    # Convert srt to segment with offset
    seg_srt = out_mp4.with_suffix(".srt")
    subs = pysubs2.load(str(in_srt))
    seg = pysubs2.SSAFile(); seg.styles = subs.styles.copy()
    for ev in subs.events:
        if ev.end <= start_ms or ev.start >= end_ms:
            continue
        new = ev.copy()
        new.start = max(0, ev.start - start_ms)
        new.end = min(end_ms - start_ms - 1, ev.end - start_ms)
        seg.events.append(new)
    seg.save(str(seg_srt))
    # Video render
    wf_h = int(h * wf_ratio)
    vf = (
        f"color=c={bg}:s={w}x{h}:r=30[bg];"
        f"[0:a]showwaves=s={w}x{wf_h}:mode=cline:rate=30:colors=White[wf];"
        f"[bg][wf]overlay=0:{h - wf_h - 120}[base];"
        f"[base]subtitles='{seg_srt.as_posix()}':force_style='FontName={font},FontSize=42,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0,Alignment=2'"
    )
    subprocess.check_call([
        "ffmpeg","-y","-i", str(out_aac),
        "-filter_complex", vf,
        "-c:v","libx264","-pix_fmt","yuv420p",
        "-c:a","copy", str(out_mp4)
    ])
    out_aac.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Auto-generate vertical clips from SRT and audio")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--outdir", default="output/clips")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--min_s", type=float, default=15.0)
    ap.add_argument("--max_s", type=float, default=40.0)
    ap.add_argument("--window_ms", type=int, default=2500)
    ap.add_argument("--filter_speaker", default="")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--bg", default="black")
    ap.add_argument("--wf_ratio", type=float, default=0.35)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    subs = pysubs2.load(args.srt)
    scored = []
    for i, ev in enumerate(subs.events):
        txt = re.sub(r"{[^}]+}", "", ev.text)
        if args.filter_speaker and not txt.lower().startswith(args.filter_speaker.lower()+":"):
            continue
        content = txt.split(":",1)[-1] if ":" in txt else txt
        s = score_line(content)
        if s > 0:
            scored.append((s, i, content.strip()))
    scored.sort(reverse=True, key=lambda x: x[0])

    used = set()
    picked = []
    min_ms = int(args.min_s*1000); max_ms = int(args.max_s*1000)
    for s, idx, _ in scored:
        if any(abs(idx - j) <= 2 for j in used):
            continue
        start, end = merge_window(subs.events, idx, args.window_ms)
        dur = end - start
        if dur < min_ms:
            end = start + min_ms
        if (end - start) > max_ms:
            end = start + max_ms
        picked.append((start, end, idx, s))
        used.update(range(idx-1, idx+2))
        if len(picked) >= args.n:
            break

    for k, (st, en, idx, sc) in enumerate(picked, 1):
        out = Path(outdir) / f"clip_{k:02d}_vertical.mp4"
        print(f"🎬 Clip {k}: {format_ts(st)}–{format_ts(en)} (score={sc:.1f})")
        render_clip(Path(args.audio), Path(args.srt), out, start_ms=st, end_ms=en, w=1080, h=1920, font=args.font, bg=args.bg, wf_ratio=args.wf_ratio)


if __name__ == "__main__":
    main()


