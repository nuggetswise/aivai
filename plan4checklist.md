## Plan 4 – Implementation Checklist (ElevenLabs TTS + A/V Outputs + Clips)

### A. Align Transcript Format with Builders
- [x] Update `md_podcast_build.py` parser to accept our headers:
  - [x] Support speaker sections: `## Alex Chen - Opening|Crossfire|Closing` (already OK)
  - [x] Support host sections: accept both `## Host - Intro` and `## Host - Intro - Alex`, and `## Host - Outro` and `## Host - Outro - Nova`
  - [x] Parse host sections as segments with `speaker = Host` and `section = Intro|Outro` (or use the actual host name if provided)
- [x] Ensure parser ignores non-speaker headings and keeps brief stage cues inline
- [ ] Validate with a sample from `data/transcripts/*_transcript.md`
 - [ ] Validate with a sample from `data/transcripts/*_transcript.md`

### B. Voice Mapping / Config
- [x] Create a config source for TTS (YAML/JSON), e.g. `config/tts_voices.yaml`:
  - [x] Map display names (`Alex Chen`, `Nova Rivers`, `Host`) to ElevenLabs voice IDs
  - [x] Allow overrides per episode/topic
- [x] Modify builder(s) to load voice map from config instead of hardcoding
- [x] Add env var `ELEVEN_API_KEY` handling + helpful error when missing

### C. ElevenLabs TTS Integration
- [ ] Implement retry/backoff on API calls; handle rate limits (429) and timeouts
- [ ] Add graceful fallback: if TTS fails, skip segment or use a local TTS placeholder and continue
- [x] Normalize loudness (e.g., target −16 LUFS-ish) consistently across segments
- [x] Insert short silences between segments (configurable)

### D. Captions (SRT/VTT)
- [x] Keep simple sentence split but guard edge cases (citations `[S#]`, stage cues)
- [x] Ensure per-segment timing distributes proportionally by word count
- [x] Emit both `.srt` and `.vtt`, write to `output/`
- [ ] Verify readability (font size, outline) on mobile

### E. Video Renders (Vertical/Square/Horizontal)
- [x] Parameterize visuals (BG color, font, waveform height)
- [x] Use ffmpeg filters to render waveform + burn-in captions
- [x] Export:
  - [x] `output/episode.mp3`
  - [x] `output/episode.srt` + `.vtt`
  - [x] `output/video_vertical.mp4` (1080×1920)
  - [x] `output/video_square.mp4` (1080×1080)
  - [x] `output/video_horizontal.mp4` (1920×1080)
- [ ] Validate on sample devices (iOS/Android) for legibility

### F. Auto-Clip Maker
- [ ] Confirm keyword heuristic aligns with the topic; make weights configurable
- [x] Add CLI flags: number of clips, min/max duration, filter by speaker
- [x] Ensure clips use same font/colors as main video; write to `output/clips/`
- [ ] Option: generate square/horizontal variants for each clip

### G. Pipeline Wiring
- [x] After `EpisodeRunner` writes JSON + Markdown, invoke builder:
  - [x] Input: Markdown path from `data/transcripts/{episode_id}_transcript.md`
  - [x] Output: capture produced file paths (mp3, srt, vtt, mp4s)
- [ ] Orchestrator emits `beats[]` and `references[]`; TTS uses cleaned `beats[].text` (no [S#]/[L#]/[R#]) for audio
- [x] Include asset paths in episode completion payload
- [x] Persist asset paths in `Episode`
- [ ] Option: background job/queue for rendering to avoid blocking

### H. Prompts & Dynamic Host Sections
- [x] Confirm host intro/outro are dynamic (already implemented from turns)
- [x] Add optional prompt templates: `app/prompts/host_intro.txt`, `app/prompts/host_outro.txt`
- [x] Variables available: `{topic, speaker_a, speaker_b, snippet_a, snippet_b, takeaway_a, takeaway_b}`

### I. DevEx / Config & Docs
- [x] Add deps to `pyproject.toml`: `requests`, `pydub`, `pysubs2`, `regex`
- [ ] Check ffmpeg presence; print actionable install guidance if missing
- [ ] Document environment and usage in `docs/lightning_setup.md` or new `docs/av_pipeline.md`
  - [ ] How to configure voices, API key, and outputs
  - [ ] One-liners to render full assets from a finished episode

### J. Validation & QA
- [ ] Golden test: run end-to-end on one episode and verify:
  - [ ] Host intro/outro included when present
  - [ ] All speaker sections rendered; citations present in transcript/captions and references metadata; markers are stripped from spoken audio
  - [ ] Audio artifacts sound natural; volumes balanced
  - [ ] Captions sync reasonably; styling legible
  - [ ] Videos render without ffmpeg errors; sizes correct
  - [ ] Clip selection produces usable hooks

### K. Frontend Integration (optional)
- [ ] Serve generated assets via existing `/files` routes
- [ ] Add links or preview in UI after episode completion
- [ ] Consider social export helpers (filename patterns, share text)

### Known Mismatches to Fix
- [x] Parser currently matches only `Opening|Crossfire|Closing`; it must also include host sections
- [x] Our headers may include extra hyphens (e.g., `## Host - Intro - Alex`); parser must accept both forms
- [x] `VOICE_MAP` should not be hardcoded; move to config


