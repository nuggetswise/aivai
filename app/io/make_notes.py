import argparse
import json
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--transcript", required=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    if args.transcript and os.path.exists(args.transcript):
        with open(args.transcript, "r", encoding="utf-8") as f:
            try:
                rows = json.load(f)
            except Exception:
                rows = []

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Debate: {args.topic}\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("## Chapters\n")
        chapters = ["opening", "positions", "crossfire", "closing"]
        for i, ch in enumerate(chapters, start=1):
            f.write(f"- {i}. {ch.title()}\n")
        f.write("\n## Transcript (with citations)\n\n")
        for r in rows:
            avatar = r.get("avatar", "?")
            phase = r.get("phase", "?")
            text = r.get("text", "")
            f.write(f"**{avatar} / {phase}**: {text}\n\n")
        f.write("\n---\n*Each factual sentence should include [S#]/[L#] citations.*\n")

    print(args.out)


if __name__ == "__main__":
    main()
