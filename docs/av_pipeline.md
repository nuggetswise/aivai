## A/V Pipeline: Voices, API Key, and Usage

### 1) Prerequisites
- FFmpeg installed and on PATH (`ffmpeg -version` should work)
- Python deps installed: `pip install -r` (or ensure pydub, pysubs2, requests, PyYAML, python-dotenv are installed)

### 2) Configure ElevenLabs
- Add your key to `.env`:
  - `ELEVEN_API_KEY=your_elevenlabs_key`
  - Optional: `ELEVEN_MODEL=eleven_multilingual_v2`
- The builders auto-load `.env` via `python-dotenv`.

### 3) Configure Voices
- Edit `config/tts_voices.yaml` and map display names to ElevenLabs voice names/UUIDs:

```yaml
voices:
  Host: "Rachel"
  Alex Chen: "Antoni"
  Nova Rivers: "Rachel"
```

- Per-episode overrides: create `config/tts_voices.{episode_id}.yaml` to override voices for that run.

### 4) Inputs: Markdown Transcript
- Use scriptfix-style Markdown from the backend (`data/transcripts/{episode_id}_transcript.md`).
- Required H2 sections: `## Name - Opening|Crossfire|Closing` (optional Intro/Outro if enabled).

### 5) Build Assets Locally

Audio + Captions + Videos (vertical/square/horizontal):
```bash
python md_podcast_build.py --md data/transcripts/{episode_id}_transcript.md \
  --slug {episode_id} \
  --voices config/tts_voices.yaml
```

- Fast run without calling TTS (uses silence placeholders):
```bash
python md_podcast_build.py --md data/transcripts/{episode_id}_transcript.md \
  --slug {episode_id} --voices config/tts_voices.yaml --no-tts
```

Outputs (in `output/`):
- `{episode_id}.mp3`
- `{episode_id}.srt` and `{episode_id}.vtt`
- `{episode_id}_vertical.mp4`, `{episode_id}_square.mp4`, `{episode_id}_horizontal.mp4`

### 6) Auto Clips (optional)
```bash
python clip_maker.py --audio output/{episode_id}.mp3 --srt output/{episode_id}.srt --n 3
```
Outputs to `output/clips/`.

### 7) Serving Assets
- Backend mounts `/media` → `output/`. Access via: `http://localhost:8000/media/*`
- Backend also returns `media_assets` in the episode completion event.

### 8) Troubleshooting
- Missing imports: `pip install pysubs2 pydub PyYAML python-dotenv`
- ffmpeg missing: install via Homebrew (`brew install ffmpeg`)
- Rate limits/timeouts: builder retries with backoff; falls back to silence if needed
- Long render: fixed by enforcing `-shortest` and `-t` equal to audio length


