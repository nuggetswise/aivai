# LangManus-style Debate Pipeline (Scaffold)

This folder contains YAML scaffolding to implement the multi-agent flow described in `app/readmelangmanus` and aligned with `plan3`.

What’s included
- pipeline.yaml: Defines the per-turn pipeline: Researcher → Commentator → Verifier → Style → TTS.
- episode_orchestrator.yaml: Defines a full-episode rundown (opening → positions → crossfire → closing), calls the per-turn pipeline, and emits a mixed audio file, transcript, and notes.

Helper scripts used by the orchestrator
- app/io/append_transcript.py: Appends a structured row to a JSON transcript for the current episode.
- app/io/mix_episode.py: Concatenates turn audio into a single episode MP3 (requires ffmpeg).
- app/io/make_notes.py: Generates Markdown show notes with chapters and the transcript.

Prerequisites
- Configure LLMs and tools in your environment (see `.env.example` and `app/config.py`). The YAML refers to logical models:
  - reasoning-llm: for persona generation (higher quality)
  - basic-llm: for researcher/verifier/style (cost-effective)
- Ensure `ffmpeg` is installed and on PATH for audio mixing.
- Dia TTS is expected to be wired in `app/tts/dia_synth.py` via the adapter; the pipeline invokes it as a local command.

Inputs/Outputs (high level)
- debate_turn (pipeline.yaml):
  - Inputs: topic, intent, phase, opponent_point, persona, local_corpus
  - Outputs: final persona-styled text with citations, audio_path
- episode_run (episode_orchestrator.yaml):
  - Inputs: topic, avatarA_path (YAML), avatarB_path (YAML), episode_id
  - Outputs: data/audio/{episode_id}.mp3, data/transcripts/{episode_id}.json, data/notes/{episode_id}.md

Mapping to your code
- Researcher → app/agents/researcher.py (uses retrieval/indexer.py, retrieval/store.py, retrieval/rank.py and io/scraper.py, io/cleaner.py)
- Commentator → app/agents/commentator.py (uses prompts/commentator.txt, persona_template.yaml)
- Verifier → app/agents/verifier.py (uses prompts/verifier.txt)
- Style → app/agents/style.py (uses persona template to adjust tone only)
- TTS → app/tts/dia_synth.py via app/tts/adapter.py

Next steps to make it runnable
- Implement `deps.py` with an LLM router that resolves "reasoning-llm" and "basic-llm" to concrete clients based on environment variables.
- Fill prompts in `app/prompts` and ensure agents call the correct tier.
- Ensure `retrieval/` pipeline embeds and indexes your local corpus; use `scripts/seed_index.py`.
- Optionally add an API entrypoint in `app/main.py` to drive `episode_run` and stream progress/events to the frontend.

Note
- The YAML is framework-agnostic. If you are not using a LangManus runner, adapt these to your orchestrator (e.g., Python state machine in `orchestrator/episode_runner.py`). The structure and data contracts here are intended to reduce wiring friction.
