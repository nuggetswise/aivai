import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--citations", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    episode_id = os.environ.get("EPISODE_ID", "ep_local")
    path = f"data/transcripts/{episode_id}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        citations = json.loads(args.citations)
    except json.JSONDecodeError:
        citations = []

    row = {
        "avatar": args.avatar,
        "phase": args.phase,
        "text": args.text,
        "citations": citations,
        "audio_path": args.audio,
    }

    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []

    data.append(row)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(path)


if __name__ == "__main__":
    main()
