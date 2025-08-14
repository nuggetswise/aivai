#!/usr/bin/env python3
"""
Build audio + captions + multi-aspect videos from scriptfix-formatted Markdown.
Integrates with ElevenLabs TTS; configurable voices and visuals.
"""

import os
import re
import json
import uuid
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from pydub import AudioSegment
import pysubs2
import yaml
import time
from dotenv import load_dotenv, find_dotenv


MD_SPEAKER_HDR = re.compile(r"^##\s+(.+?)\s+-\s+(Intro|Opening|Crossfire|Closing|Outro)\s*(?:-\s*.*)?$", re.I)
STAGE_CUE = re.compile(r"\((?:pause|chuckle|scoff|thoughtful|serious tone|laughs)\)", re.I)


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_md_lines(md: str) -> str:
    bad = [
        "Pipeline failed", "refined commentary", "preservation rules",
        "The current evidence doesn't support a definitive position on this.",
    ]
    lines = [ln for ln in md.splitlines() if not any(b in ln for b in bad)]
    return "\n".join(lines)


def parse_segments(md: str):
    blocks = []
    current = None
    buf: List[str] = []
    for ln in md.splitlines():
        m = MD_SPEAKER_HDR.match(ln.strip())
        if m:
            if current and buf:
                blocks.append({**current, "text": "\n".join(buf).strip()})
                buf = []
            current = {"speaker": m.group(1).strip(), "section": m.group(2).strip().title()}
        else:
            if current:
                buf.append(ln)
    if current and buf:
        blocks.append({**current, "text": "\n".join(buf).strip()})

    for b in blocks:
        text = "\n".join(t for t in b["text"].splitlines() if t.strip())
        text = re.sub(r"\s+", " ", text).strip()
        b["text"] = text
    return blocks


def load_voice_map(path: Path) -> Dict[str, str]:
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        voices = data.get("voices") or {}
        return {str(k): str(v) for k, v in voices.items()}
    return {}


def eleven_tts(text: str, voice_id: str, api_key: str, model: str = "eleven_multilingual_v2") -> AudioSegment:
    if not api_key:
        raise RuntimeError("ELEVEN_API_KEY not set")
    base = os.getenv("ELEVEN_BASE", "https://api.elevenlabs.io/v1")
    url = f"{base}/text-to-speech/{voice_id}"
    # Strip inline citations like [S1]/[L2]/[R3] from spoken text
    clean_text = re.sub(r"\[(?:S|L|R)\d+\]", "", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    payload = {"text": clean_text, "model_id": model, "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}}
    headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
    backoff = 1.0
    for attempt in range(4):
        try:
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if r.status_code == 429:
                raise RuntimeError("rate_limited")
            r.raise_for_status()
            tmp = Path("data/tmp"); tmp.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp / f"{uuid.uuid4().hex}.mp3"
            tmp_path.write_bytes(r.content)
            return AudioSegment.from_file(tmp_path, format="mp3")
        except Exception:
            if attempt == 3:
                # graceful fallback: short silence placeholder
                return AudioSegment.silent(duration=800)
            time.sleep(backoff)
            backoff *= 2


def split_sentences(text: str) -> List[str]:
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [x.strip() for x in s if x.strip()]


def strip_citations(text: str) -> str:
    return re.sub(r"\[(?:S|L|R)\d+\]", "", text)


def build_subs(segments, audio_segments, out_srt: Path, out_vtt: Path, font: str):
    subs = pysubs2.SSAFile()
    t0 = 0
    for (seg_audio, speaker, section), seg in zip(audio_segments, segments):
        dur = len(seg_audio)
        sentences = split_sentences(strip_citations(seg["text"])) or [strip_citations(seg["text"])]
        words = [max(1, len(s.split())) for s in sentences]
        total_words = sum(words) or 1
        acc = 0
        for s, w in zip(sentences, words):
            chunk = int(dur * (w / total_words))
            start = t0 + acc
            end = min(t0 + acc + chunk, t0 + dur - 1)
            ev = pysubs2.SSAEvent(start=start, end=end, text=f"{speaker}: {s}")
            subs.events.append(ev)
            acc += chunk
        t0 += dur

    style = pysubs2.SSAStyle()
    style.fontname = font
    style.fontsize = 42
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)
    style.backcolor = pysubs2.Color(0, 0, 0, 128)
    style.outline = 2
    style.shadow = 0
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    subs.styles["Default"] = style

    out_srt.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(out_srt))
    subs.save(str(out_vtt), format_="vtt")


def render_video(audio_path: Path, srt_path: Path, out_path: Path, size=(1080, 1920), font="Arial", bg="black", wf_ratio=0.35, duration_ms: int | None = None):
    w, h = size
    wf_h = int(h * wf_ratio)
    vf = (
        f"color=c={bg}:s={w}x{h}:r=30[bg];"
        f"[0:a]showwaves=s={w}x{wf_h}:mode=cline:rate=30:colors=White[wf];"
        f"[bg][wf]overlay=0:{h - wf_h - 120}[base];"
        f"[base]subtitles='{srt_path.as_posix()}':force_style='FontName={font},FontSize=42,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1,Outline=2,Shadow=0,Alignment=2'"
    )
    cmd = [
        "ffmpeg","-y",
        "-i", str(audio_path),
        "-filter_complex", vf,
        "-shortest",
        "-c:v","libx264","-pix_fmt","yuv420p",
        "-c:a","aac","-b:a","192k",
    ]
    if duration_ms is not None:
        cmd.extend(["-t", f"{duration_ms/1000:.3f}"])
    cmd.append(str(out_path))
    subprocess.check_call(cmd)


def have_ffmpeg() -> bool:
    try:
        subprocess.check_output(["ffmpeg", "-version"])
        return True
    except Exception:
        return False


def estimate_speech_ms(text: str) -> int:
    # Approx 160 wpm => ~375 ms per word
    words = max(1, len(text.split()))
    return int(min(12000, words * 375))


def main(md_file: str, episode_slug: str, voices_cfg: Path, font: str, bg: str, wf_ratio: float, no_tts: bool = False):
    # Ensure .env is loaded so ELEVEN_API_KEY is available
    load_dotenv(find_dotenv())
    md_path = Path(md_file)
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    cleaned = clean_md_lines(load_markdown(md_path))
    segments = parse_segments(cleaned)
    if not segments:
        raise SystemExit("No speaker segments found. Ensure H2 headers like '## Name - Opening' exist.")

    # allow per-episode override: config/tts_voices.{slug}.yaml
    override_cfg = Path(f"config/tts_voices.{episode_slug}.yaml")
    voice_map = load_voice_map(override_cfg if override_cfg.exists() else voices_cfg)
    api_key = os.getenv("ELEVEN_API_KEY", "")
    model = os.getenv("ELEVEN_MODEL", "eleven_multilingual_v2")

    audio_parts = []
    for seg in segments:
        speaker = seg["speaker"]
        # Host sections map to "Host" if not explicitly mapped
        key = speaker if speaker in voice_map else ("Host" if seg["section"] in {"Intro", "Outro"} else speaker)
        voice_id = voice_map.get(key)
        if no_tts:
            dur_ms = estimate_speech_ms(seg["text"]) + 400
            speech = AudioSegment.silent(duration=dur_ms)
            print(f"TTS: {speaker} / {seg['section']} (silent placeholder {dur_ms} ms)")
        else:
            if not voice_id:
                raise SystemExit(f"No voice configured for speaker: {speaker} (key: {key})")
            print(f"TTS: {speaker} / {seg['section']} …")
            speech = eleven_tts(seg["text"], voice_id, api_key=api_key, model=model)
        audio_parts.append((speech, speaker, seg["section"]))

    final = AudioSegment.silent(duration=300)
    for speech, _, _ in audio_parts:
        final += speech + AudioSegment.silent(duration=150)
    change = -16 - final.dBFS if final.dBFS != float("-inf") else 0
    final = final.apply_gain(change)

    mp3_path = out_dir / f"{episode_slug}.mp3"
    final.export(mp3_path, format="mp3", bitrate="192k")
    print(f"✅ Wrote {mp3_path}")

    srt_path = out_dir / f"{episode_slug}.srt"
    vtt_path = out_dir / f"{episode_slug}.vtt"
    build_subs(segments, audio_parts, srt_path, vtt_path, font=font)
    print(f"✅ Wrote {srt_path} and {vtt_path}")

    if have_ffmpeg():
        for name, size in {"vertical": (1080, 1920), "square": (1080, 1080), "horizontal": (1920, 1080)}.items():
            out_vid = out_dir / f"{episode_slug}_{name}.mp4"
            print(f"🎬 Rendering {name} {size[0]}x{size[1]} …")
            render_video(mp3_path, srt_path, out_vid, size=size, font=font, bg=bg, wf_ratio=wf_ratio, duration_ms=len(final))
            print(f"✅ Wrote {out_vid}")
    else:
        print("⚠️ ffmpeg not found; skipping video renders. Install ffmpeg and re-run to generate videos.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True, help="Path to cleaned Markdown")
    ap.add_argument("--slug", default="episode", help="Output filename stem")
    ap.add_argument("--voices", default="config/tts_voices.yaml", help="Path to voices YAML")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--bg", default="black")
    ap.add_argument("--wf_ratio", type=float, default=0.35)
    ap.add_argument("--no-tts", action="store_true", help="Skip TTS and use silence placeholders for speed")
    args = ap.parse_args()
    main(args.md, args.slug, Path(args.voices), args.font, args.bg, args.wf_ratio, no_tts=args["no_tts"] if isinstance(args, dict) else args.no_tts)


