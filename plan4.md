Awesome — let’s switch to **hosted TTS with ElevenLabs** and auto-make audio + three social-ready videos from your Markdown. Here’s a small, copy-paste script you can run locally.

---

# 1) `md_podcast_build.py` — from Markdown → MP3 + VTT/SRT + vertical/square/horizontal videos

**What it does**

* Parses your `.md` into segments per speaker
* Calls **ElevenLabs TTS** for each segment (speaker-specific voices)
* Concats audio, builds **WebVTT + SRT** captions (rough but solid)
* Uses **ffmpeg** filters to render a waveform + burn-in captions
* Exports:

  * `output/episode.mp3`
  * `output/episode.vtt` + `output/episode.srt`
  * `output/video_vertical.mp4` (1080×1920)
  * `output/video_square.mp4` (1080×1080)
  * `output/video_horizontal.mp4` (1920×1080)

> Install deps first:
> `pip install requests pydub pysubs2 regex` and have **ffmpeg** on PATH.

```python
#!/usr/bin/env python3
# md_podcast_build.py
import os, re, json, uuid, math, argparse, subprocess, tempfile, textwrap
from pathlib import Path
import requests
from pydub import AudioSegment
import pysubs2

# ----------------------------
# CONFIG
# ----------------------------
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
# Map speaker names (as they appear in MD headers) to Eleven voice IDs
VOICE_MAP = {
    "Host": "Rachel",        # example: "Rachel" or a voice UUID
    "Narrator": "Rachel",
    "Alex Chen": "Antoni",
    "Nova Rivers": "Rachel",
}

# Aspect presets: (w, h)
ASPECTS = {
    "vertical":   (1080, 1920),
    "square":     (1080, 1080),
    "horizontal": (1920, 1080),
}

BG_COLOR = "black"   # solid background color for ffmpeg color source
FONT = "Arial"       # libass font for subtitles (installed on your system)
WF_HEIGHT_RATIO = 0.35  # waveform height relative to video height

ELEVEN_MODEL = "eleven_multilingual_v2"  # solid default
ELEVEN_BASE = "https://api.elevenlabs.io/v1"

# ----------------------------
# Markdown parsing
# ----------------------------
MD_SPEAKER_HDR = re.compile(r"^##\s+(.+?)\s+-\s+(Opening|Crossfire|Closing)\s*$", re.I)
STAGE_CUE = re.compile(r"\((?:pause|chuckle|scoff|thoughtful|serious tone|laughs)\)", re.I)

def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def clean_md_lines(md: str) -> str:
    # Remove pipeline artifacts & duplicate filler lines
    bad = [
        "Pipeline failed", "refined commentary", "preservation rules",
        "The current evidence doesn't support a definitive position on this." # typical duplicate
    ]
    lines = [ln for ln in md.splitlines() if not any(b in ln for b in bad)]
    return "\n".join(lines)

def parse_segments(md: str):
    """
    Returns list of dicts: {speaker, section, text}
    Splits by H2 like: "## Alex Chen - Opening"
    """
    blocks = []
    current = None
    buf = []
    for ln in md.splitlines():
        m = MD_SPEAKER_HDR.match(ln.strip())
        if m:
            # flush previous
            if current and buf:
                blocks.append({**current, "text": "\n".join(buf).strip()})
                buf = []
            current = {"speaker": m.group(1).strip(), "section": m.group(2).strip()}
        else:
            if current:
                buf.append(ln)
    if current and buf:
        blocks.append({**current, "text": "\n".join(buf).strip()})

    # Strip extra blank lines and keep short stage cues
    for b in blocks:
        text = "\n".join(t for t in b["text"].splitlines() if t.strip())
        # collapse multiple spaces; keep short stage cues in-line
        text = re.sub(r"\s+", " ", text).strip()
        b["text"] = text
    return blocks

# ----------------------------
# ElevenLabs TTS
# ----------------------------
def eleven_tts(text: str, voice_id: str) -> AudioSegment:
    if not ELEVEN_API_KEY:
        raise RuntimeError("ELEVEN_API_KEY not set")
    url = f"{ELEVEN_BASE}/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": ELEVEN_MODEL,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}
    }
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json"
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(r.content); tmp.close()
    return AudioSegment.from_file(tmp.name, format="mp3")

# ----------------------------
# Captions (SRT + VTT)
# ----------------------------
def split_sentences(text: str):
    # simple sentence split (avoid heavy NLP deps)
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [x.strip() for x in s if x.strip()]

def build_subs(segments, audio_segments, out_srt: Path, out_vtt: Path):
    """
    segments: list of dicts {speaker, section, text}
    audio_segments: list of (AudioSegment, speaker, section)
    - We assign per-sentence durations proportionally by word count within each segment.
    """
    subs = pysubs2.SSAFile()
    t0 = 0  # ms
    for (seg_audio, speaker, section), seg in zip(audio_segments, segments):
        dur = len(seg_audio)  # ms
        sentences = split_sentences(seg["text"])
        words = [max(1, len(s.split())) for s in sentences]
        total_words = sum(words) or 1
        acc = 0
        for s, w in zip(sentences, words):
            chunk = int(dur * (w / total_words))
            start = t0 + acc
            end = min(t0 + acc + chunk, t0 + dur - 1)
            ev = pysubs2.SSAEvent(
                start=start, end=end,
                text=f"{speaker}: {s}"
            )
            subs.events.append(ev)
            acc += chunk
        t0 += dur

    # Style for burn-in (libass)
    style = pysubs2.SSAStyle()
    style.fontname = FONT
    style.fontsize = 42
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)   # white
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)         # black outline
    style.backcolor = pysubs2.Color(0, 0, 0, 128)          # semi-transparent box
    style.bold = False
    style.outline = 2
    style.shadow = 0
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    subs.styles["Default"] = style

    out_srt.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(out_srt))        # SRT
    subs.save(str(out_vtt), format_="vtt")  # VTT

# ----------------------------
# Waveform + captions video via ffmpeg
# ----------------------------
def render_video(audio_path: Path, srt_path: Path, out_path: Path, size=(1080,1920)):
    w, h = size
    # waveform height
    wf_h = int(h * WF_HEIGHT_RATIO)
    # Filters:
    #  - color background
    #  - showwaves (from audio) as line
    #  - position waveform in lower third
    #  - burn subtitles with libass (SRT auto-converted by ffmpeg)
    vf = (
        f"color=c={BG_COLOR}:s={w}x{h}:r=30[bg];"
        f"[0:a]showwaves=s={w}x{wf_h}:mode=cline:rate=30:colors=White[wf];"
        f"[bg][wf]overlay=0:{h - wf_h - 120}[base];"
        f"[base]subtitles='{srt_path.as_posix()}':force_style="
        f"'FontName={FONT},FontSize=42,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0,Alignment=2'"
    )
    cmd = [
        "ffmpeg","-y",
        "-i", str(audio_path),
        "-filter_complex", vf,
        "-c:v","libx264","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k",
        str(out_path)
    ]
    subprocess.check_call(cmd)

# ----------------------------
# Main
# ----------------------------
def main(md_file: str, episode_slug: str):
    md_path = Path(md_file)
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    cleaned = clean_md_lines(load_markdown(md_path))
    segments = parse_segments(cleaned)
    if not segments:
        raise SystemExit("No speaker segments found. Ensure H2 headers like '## Alex Chen - Opening' exist.")

    # TTS each segment
    audio_parts = []
    for seg in segments:
        speaker = seg["speaker"]
        voice_id = VOICE_MAP.get(speaker) or VOICE_MAP.get("Narrator")
        if not voice_id:
            raise SystemExit(f"No voice configured for speaker: {speaker}")
        print(f"TTS: {speaker} / {seg['section']} …")
        speech = eleven_tts(seg["text"], voice_id)
        audio_parts.append((speech, speaker, seg["section"]))

    # Mixdown
    final = AudioSegment.silent(duration=300)  # 0.3s lead-in
    for speech, _, _ in audio_parts:
        final += speech + AudioSegment.silent(duration=150)  # 150ms gap
    # Loudness-ish normalize
    change = -16 - final.dBFS if final.dBFS != float("-inf") else 0
    final = final.apply_gain(change)

    mp3_path = out_dir / f"{episode_slug}.mp3"
    final.export(mp3_path, format="mp3", bitrate="192k")
    print(f"✅ Wrote {mp3_path}")

    # Subs
    srt_path = out_dir / f"{episode_slug}.srt"
    vtt_path = out_dir / f"{episode_slug}.vtt"
    build_subs(segments, audio_parts, srt_path, vtt_path)
    print(f"✅ Wrote {srt_path} and {vtt_path}")

    # Videos
    for name, (W,H) in ASPECTS.items():
        out_vid = out_dir / f"video_{name}.mp4"
        print(f"🎬 Rendering {name} {W}x{H} …")
        render_video(mp3_path, srt_path, out_vid, size=(W,H))
        print(f"✅ Wrote {out_vid}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="Path to cleaned Markdown")
    ap.add_argument("--slug", default="episode", help="Output filename stem")
    args = ap.parse_args()
    main(args.md, args.slug)
```

**Usage**

```bash
export ELEVEN_API_KEY=xxxxxxxxxxxxxxxxxxxx
python md_podcast_build.py --md grok_debate.md --slug grok-aurora
```

> Adjust `VOICE_MAP` to your ElevenLabs voices (use names or UUIDs).
> If you also want an **intro/outro narrator**, leave a `## Host - Intro` and `## Host - Outro` section in the MD; the parser will include them automatically.



---

# `clip_maker.py` — auto-select hooks → render vertical clips

**Installs:**

```bash
pip install pysubs2 regex
# (ffmpeg must be on PATH)
```

```python
#!/usr/bin/env python3
# clip_maker.py
import argparse, subprocess, re, os
from pathlib import Path
import pysubs2

# Heuristic: keyword weights (tweak freely)
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
    r"\bAurora\b|\bGrok\b": 1.0,
    r"[?!]": 0.8,
}

def score_line(text:str)->float:
    s = 0.0
    low = text.lower()
    for pat, w in KEYWORDS.items():
        if re.search(pat, low):
            s += w
    # prefer 6–20 words; penalize too short/long
    n = len(low.split())
    if 6 <= n <= 20: s += 1.2
    elif n < 4: s -= 0.8
    elif n > 36: s -= 0.6
    return s

def merge_window(events, idx, window_ms):
    """Return (start_ms, end_ms, text_joined) spanning neighboring lines within window."""
    start = events[idx].start
    end   = events[idx].end
    text  = [events[idx].text]
    # extend backward
    i = idx-1
    while i>=0 and (start - events[i].end) <= window_ms:
        start = min(start, events[i].start)
        text.insert(0, events[i].text)
        i-=1
    # extend forward
    j = idx+1
    while j<len(events) and (events[j].start - end) <= window_ms:
        end = max(end, events[j].end)
        text.append(events[j].text)
        j+=1
    return start, end, " ".join(text)

def clamp_duration(start_ms, end_ms, min_ms, max_ms, total_ms):
    dur = end_ms - start_ms
    if dur < min_ms:
        pad = (min_ms - dur)//2
        start_ms = max(0, start_ms - pad)
        end_ms   = min(total_ms-1, end_ms + pad + (min_ms - (end_ms - start_ms)))
    if (end_ms - start_ms) > max_ms:
        end_ms = start_ms + max_ms
    return start_ms, end_ms

def format_ts(ms):
    s, ms = divmod(ms, 1000)
    m, s  = divmod(s, 60)
    h, m  = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def cut_srt_segment(in_srt:Path, out_srt:Path, start_ms:int, end_ms:int):
    subs = pysubs2.load(str(in_srt))
    seg  = pysubs2.SSAFile()
    seg.styles = subs.styles.copy()
    for ev in subs.events:
        # keep overlap
        if ev.end <= start_ms or ev.start >= end_ms:
            continue
        new = ev.copy()
        new.start = max(0, ev.start - start_ms)
        new.end   = min(end_ms - start_ms - 1, ev.end - start_ms)
        seg.events.append(new)
    seg.save(str(out_srt))

def render_clip(in_audio:Path, in_srt:Path, out_mp4:Path, start_ms:int, end_ms:int, w=1080, h=1920,
                font="Arial", bg="black", wf_ratio=0.35):
    # trim audio
    out_aac = out_mp4.with_suffix(".aac")
    ss = format_ts(start_ms)
    to = format_ts(end_ms - start_ms)
    subprocess.check_call([
        "ffmpeg","-y","-ss", ss, "-t", to, "-i", str(in_audio),
        "-c:a","aac","-b:a","192k", str(out_aac)
    ])
    # cut srt
    seg_srt = out_mp4.with_suffix(".srt")
    cut_srt_segment(in_srt, seg_srt, start_ms, end_ms)
    # video filters
    wf_h = int(h*wf_ratio)
    vf = (
        f"color=c={bg}:s={w}x{h}:r=30[bg];"
        f"[0:a]showwaves=s={w}x{wf_h}:mode=cline:rate=30:colors=White[wf];"
        f"[bg][wf]overlay=0:{h - wf_h - 120}[base];"
        f"[base]subtitles='{seg_srt.as_posix()}':force_style="
        f"'FontName={font},FontSize=42,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,"
        f"BorderStyle=1,Outline=2,Shadow=0,Alignment=2'"
    )
    subprocess.check_call([
        "ffmpeg","-y","-i", str(out_aac),
        "-filter_complex", vf,
        "-c:v","libx264","-pix_fmt","yuv420p",
        "-c:a","copy", str(out_mp4)
    ])
    os.remove(out_aac)
    # keep seg_srt for review

def main():
    ap = argparse.ArgumentParser(description="Auto-make highlight clips from SRT")
    ap.add_argument("--audio", required=True, help="Full episode MP3/WAV")
    ap.add_argument("--srt", required=True, help="Full episode SRT")
    ap.add_argument("--outdir", default="output/clips")
    ap.add_argument("--n", type=int, default=3, help="number of clips")
    ap.add_argument("--min_s", type=float, default=15.0)
    ap.add_argument("--max_s", type=float, default=40.0)
    ap.add_argument("--window_ms", type=int, default=2500, help="merge neighbor subs within this window")
    ap.add_argument("--filter_speaker", default="", help="only keep lines starting with 'Speaker:'")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    subs = pysubs2.load(args.srt)
    total_ms = int(subs.events[-1].end) if subs.events else 0

    # Score lines
    scored = []
    for i, ev in enumerate(subs.events):
        txt = re.sub(r"{[^}]+}", "", ev.text)  # strip tags
        # Expect format "Speaker: sentence"
        if args.filter_speaker and not txt.lower().startswith(args.filter_speaker.lower()+":"):
            continue
        # score only the sentence content
        content = txt.split(":",1)[-1] if ":" in txt else txt
        s = score_line(content)
        if s > 0:
            scored.append((s, i, content.strip()))

    scored.sort(reverse=True, key=lambda x: x[0])

    picked = []
    used = set()
    min_ms = int(args.min_s*1000); max_ms = int(args.max_s*1000)

    for s, idx, _ in scored:
        # avoid overlapping with already picked
        if any(abs(idx - j) <= 2 for j in used):  # crude overlap guard
            continue
        start, end, _ = merge_window(subs.events, idx, args.window_ms)
        start, end = clamp_duration(start, end, min_ms, max_ms, total_ms)
        picked.append((start, end, idx, s))
        used.update(range(idx-1, idx+2))
        if len(picked) >= args.n:
            break

    if not picked:
        print("No hooky lines found. Try loosening keywords or lowering --min_s")
        return

    for k, (st, en, idx, sc) in enumerate(picked, 1):
        out = outdir / f"clip_{k:02d}_vertical.mp4"
        print(f"🎬 Clip {k}: {format_ts(st)}–{format_ts(en)} (score={sc:.1f})")
        render_clip(Path(args.audio), Path(args.srt), out,
                    start_ms=st, end_ms=en, w=1080, h=1920)

if __name__ == "__main__":
    main()
```

**Usage**

```bash
# Make 3 clips (vertical) from your full assets:
python clip_maker.py --audio output/grok-aurora.mp3 --srt output/grok-aurora.srt --n 3

# Only lines spoken by Nova Rivers:
python clip_maker.py --audio output/grok-aurora.mp3 --srt output/grok-aurora.srt --filter_speaker "Nova Rivers"
```

> If you also want square/horizontal from the same timecodes, you can call `render_clip(...)` again with different `w,h` or add a loop similar to the main video script.

can fold `clip_maker.py` into your **md\_podcast\_build.py** so it auto-generates clips right after the main episode renders, using the same fonts and output folder.
