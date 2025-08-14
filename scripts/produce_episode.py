#!/usr/bin/env python3
"""
One-command episode producer.

Pipeline:
1) Generate debate JSON via scripts/test_conversation.py
2) Convert JSON → scriptfix Markdown via scripts/generate_transcript.py
3) Build audio/captions/videos via md_podcast_build.py

Usage example:
  python scripts/produce_episode.py \
    --topic "Grok recent image generation" \
    --avatarA avatars/alex.yaml --avatarB avatars/nova.yaml \
    --tighten --include-host --no-tts
"""

import argparse
import subprocess
from pathlib import Path
import sys
import shlex


def snake(s: str) -> str:
    return "_".join(s.strip().lower().split())


def run(cmd: list[str]):
    print("$", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Produce a full episode from topic → JSON → Markdown → A/V")
    ap.add_argument("--topic", required=True, help="Debate topic")
    ap.add_argument("--avatarA", default="avatars/alex.yaml")
    ap.add_argument("--avatarB", default="avatars/nova.yaml")
    ap.add_argument("--include-host", action="store_true", help="Include intro/outro sections in Markdown")
    ap.add_argument("--tighten", action="store_true", help="Tighten paragraphs to 2–3 sentences")
    ap.add_argument("--max-sentences", type=int, default=3)
    ap.add_argument("--keep-filler", action="store_true")
    ap.add_argument("--no-tts", action="store_true", help="Skip ElevenLabs TTS; use silence placeholders")
    ap.add_argument("--voices", default="config/tts_voices.yaml")
    ap.add_argument("--slug", default="", help="Output filename stem (default: derived from topic)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # Derive paths
    slug = args.slug or snake(args.topic)
    json_path = project_root / "data" / "transcripts" / f"debate_{snake(args.topic)}.json"
    md_path = project_root / "data" / "transcripts" / f"{slug}.md"

    # 1) Generate debate JSON
    run([
        sys.executable,
        str(project_root / "scripts" / "test_conversation.py"),
        "--topic", args.topic,
        "--alex", args.avatarA,
        "--nova", args.avatarB,
    ])

    if not json_path.exists():
        print(f"Error: transcript JSON not found at {json_path}", file=sys.stderr)
        sys.exit(2)

    # 2) Convert JSON → Markdown
    gen_cmd = [
        sys.executable,
        str(project_root / "scripts" / "generate_transcript.py"),
        "--input", str(json_path),
        "--output", str(md_path),
    ]
    if args.tighten:
        gen_cmd.append("--tighten")
        gen_cmd.extend(["--max-sentences", str(args.max_sentences)])
        if args.keep_filler:
            gen_cmd.append("--keep-filler")
    if args.include_host:
        gen_cmd.append("--include-host")
    run(gen_cmd)

    if not md_path.exists():
        print(f"Error: Markdown not found at {md_path}", file=sys.stderr)
        sys.exit(3)

    # 3) Build media assets
    build_cmd = [
        sys.executable,
        str(project_root / "md_podcast_build.py"),
        "--md", str(md_path),
        "--slug", slug,
        "--voices", args.voices,
    ]
    if args.no_tts:
        build_cmd.append("--no-tts")
    run(build_cmd)

    out_dir = project_root / "output"
    print("\nAssets:")
    print("  ", out_dir / f"{slug}.mp3")
    print("  ", out_dir / f"{slug}.srt")
    print("  ", out_dir / f"{slug}.vtt")
    print("  ", out_dir / f"{slug}_vertical.mp4")
    print("  ", out_dir / f"{slug}_square.mp4")
    print("  ", out_dir / f"{slug}_horizontal.mp4")


if __name__ == "__main__":
    main()


