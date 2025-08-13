import argparse
import subprocess
import os
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated list of audio file paths")
    parser.add_argument("--out", required=True, help="Output mp3 path")
    args = parser.parse_args()

    inputs = [p for p in args.inputs.split(",") if p]
    if not inputs:
        raise SystemExit("No input audio segments provided")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Build ffconcat file for clean joining
    concat_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(concat_file, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for p in inputs:
            abspath = os.path.abspath(p)
            if not os.path.exists(abspath):
                raise SystemExit(f"Missing input audio segment: {abspath}")
            f.write(f"file '{abspath}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "mp3",
        args.out,
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        raise SystemExit("ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")

    print(args.out)


if __name__ == "__main__":
    main()
