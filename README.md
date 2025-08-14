# AI vs AI Podcast Platform

A powerful platform for creating AI-driven debate podcasts between AI avatars with distinct personas, evidence-based arguments, and high-quality audio synthesis.

## Overview

AI vs AI is an advanced podcast generation platform that creates compelling debate content through AI avatars with distinct personalities. Each avatar leverages its own research corpus to develop evidence-based arguments on debate topics, with automatic fact-checking, citation enforcement, and high-quality text-to-speech synthesis.

![AI vs AI Podcast System Architecture](https://via.placeholder.com/800x400?text=AI+vs+AI+Architecture)

## Key Features

- **Distinct AI Personas**: Create AI avatars with configurable personalities, speech patterns, and knowledge bases
- **Evidence-Based Arguments**: Enforced citation requirements and fact verification
- **Multi-Phase Debates**: Structured debates with opening statements, rebuttals, and closing arguments
- **High-Quality Audio**: TTS synthesis with audio mixing and normalization
- **Web-Based UI**: Control panel for managing episodes and avatars
- **Extensible Architecture**: Modular design for adding new agents and capabilities

## System Architecture

The platform consists of several key components:

1. **Orchestrator**: Manages the debate workflow and avatar interactions
2. **Agents**: Specialized AI agents for research, commentary, verification, and style
3. **Retrieval System**: Fetches and indexes relevant information from trusted sources
4. **TTS System**: Converts text to high-quality synthesized speech
5. **API Layer**: REST API for interacting with the system
6. **Frontend**: Web UI for managing debates and avatars

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg
- Node.js 18+ (for frontend)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-vs-ai-podcast.git
   cd ai-vs-ai-podcast
   ```

2. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

3. Install dependencies:
   ```bash
   make install
   ```

### A/V Pipeline

See `docs/av_pipeline.md` for configuring ElevenLabs voices and building audio/captions/videos.

### Running Your First Episode

1. Create or use the example avatars:
   ```bash
   # Example avatars are in avatars/skeptic.yaml and avatars/poppulse.yaml
   ```

2. Seed the knowledge base for avatars:
   ```bash
   make seed-avatar AVATAR=avatars/skeptic.yaml SOURCES=corpus/skeptic/links.csv
   make seed-avatar AVATAR=avatars/poppulse.yaml SOURCES=corpus/poppulse/links.csv
   ```

3. Run a debate episode:
   ```bash
   make episode TOPIC="AI Safety and Regulation" AVATAR_A=avatars/skeptic.yaml AVATAR_B=avatars/poppulse.yaml
   ```

4. Find the output in:
   - Audio: `data/audio/{episode_id}.mp3`
   - Transcript: `data/transcripts/{episode_id}.json`
   - Notes: `data/notes/{episode_id}.md`

### Web Interface

Start the web interface:
```bash
make run
```

Visit `http://localhost:8000` to access the podcast producer interface.

## Avatar Configuration

Avatars are defined using YAML files with the following structure:

```yaml
name: "Skeptic"
role: "Tech Ethicist"
tone: "calm, precise"
speech_quirks:
  - "uses historical analogies"
  - "asks short, probing questions"
default_unknown: "I don't have sufficient evidence to comment on that."
voice:
  vendor: "dia"
  ref_audio: "data/voices/skeptic_ref.wav"
  speaker_tag: "[S1]"
```

### Avatar Corpus

Each avatar has an associated corpus configuration that defines its knowledge sources:

```yaml
avatar_name: "Skeptic"
avatar_path: "avatars/skeptic.yaml"
sources:
  academic: 45     # Percentage of academic sources
  news: 20         # Percentage of news sources
  tech: 20         # Percentage of tech blogs
  historical: 10   # Percentage of historical archives
preferred_domains:
  - domain: "*.edu"
    weight: 10
  - domain: "nature.com"
    weight: 9
topics:
  - "AI Safety"
  - "Machine Learning"
  - "Technology Ethics"
biases:
  - "Prefers evidence-based arguments over appeals to emotion"
  - "Tends to cite academic sources more frequently"
```

## System Components

### Agents

- **Researcher**: Gathers evidence from web and local corpus
- **Commentator**: Generates persona-locked debate turns
- **Verifier**: Validates factual claims against evidence
- **Style**: Applies persona tone and quirks without altering facts

### IO Utilities

- **Scraper**: Web content fetching and extraction
- **Cleaner**: Text normalization and cleaning
- **Files**: File management for transcripts, bundles, etc.
- **Audio**: Audio normalization, mixing, and format conversion

### Orchestration

The debate follows a structured flow:

1. **Research Phase**: Both avatars gather evidence
2. **Opening Phase**: Each avatar presents their initial position
3. **Debate Phase**: Alternating rebuttals and arguments
4. **Closing Phase**: Final arguments and summary

## Content Policies

Source content is filtered based on whitelist/greylist/blacklist policies defined in `whitelist.yaml`. These policies ensure content quality and factual reliability.

## Extending the Platform

### Adding New Avatars

1. Create a new avatar YAML in `avatars/{name}.yaml`
2. Create a corpus directory in `corpus/{name}/`
3. Define a manifest in `corpus/{name}/manifest.yaml`
4. Add a reference voice file in `data/voices/{name}_ref.wav`
5. Seed the avatar's knowledge base with `make seed-avatar`

### Custom TTS Voices

The platform supports multiple TTS providers:

- DIA TTS (default)
- Gemini-only stack for LLM; OpenAI disabled
- ElevenLabs
- Custom providers through the adapter interface

## Advanced Usage

### Command Line Interface

```bash
# Run debate with custom parameters
python scripts/run_episode.py --topic "Climate Change Solutions" --avatar-a avatars/skeptic.yaml --avatar-b avatars/poppulse.yaml --turns 3

# Custom knowledge seeding
python scripts/seed_index.py --avatar avatars/skeptic.yaml --sources corpus/skeptic/links.csv --max 50 --freshness 90
```

### API

- Live docs: `http://localhost:8000/docs` (Swagger) and `http://localhost:8000/redoc`
- Reference: `docs/api.md` (endpoints and request/response examples)

## Development

### Running Tests

```bash
make test
```

### Directory Structure

```
ai-vs-ai-podcast/
├── app/                  # Core application code
│   ├── agents/           # AI agents for various tasks
│   ├── io/               # Input/output utilities
│   ├── orchestrator/     # Debate flow management
│   ├── retrieval/        # Web retrieval and indexing
│   └── tts/              # Text-to-speech modules
├── avatars/              # Avatar definitions
├── corpus/               # Knowledge corpus for avatars
├── data/                 # Generated data (audio, transcripts)
├── frontend/             # Web UI
└── scripts/              # Utility scripts
```

## License

[License information goes here]

## Acknowledgments

- [Any acknowledgments or credits]
