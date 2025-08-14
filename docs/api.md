## AIvAI API Reference

- Base URL: http://127.0.0.1:8000
- Interactive docs: /docs (Swagger UI), /redoc

### Health
- GET /health → { ok, ffmpeg, env.{ELEVEN_API_KEY,GEMINI_API_KEY,TAVILY_API_KEY} }

### Episodes
- POST /episodes → { episode_id, status, message }
- POST /episodes/{id}/start → { message, episode_id }
- GET /episodes/{id}/status → EpisodeStatus
- GET /episodes/{id} → Episode
- GET /episodes/{id}/assets → { episode_id, assets.{ mp3,srt,vtt,videos[],clips[] } }
- WS /ws/{id} → events: status_update, phase_change, turn_generated, turn_complete, episode_complete

### Files
- GET /files/output/{path} → serve any file from output/
- Static mounts: /media → output/, /static → data/

### Research & Knowledge
- POST /research
  - Body: { topic: string, perspective?: "pro"|"con", max_results?: number }
  - Resp: { success, data.{ keyPoints[], sources[], perspective } }
- POST /knowledge/process
  - Body: { avatarId: string, topic: string, researchResults: [{ title, content, url? }] }
  - Resp: { success, data.{ corpusId, chunksCount, summary, keyPoints[], processedAt } }
- POST /knowledge/search
  - Body: { avatarId: string, query: string, maxResults?: number }
  - Resp: { success, data.{ query, results:[{ id, content, source, url?, similarity }], totalResults } }

### Debate
- POST /debate/generate
  - Body: { topic, avatars:[{ id,name,stance,personality,knowledge[] }], avatarId, phase?: "opening"|"main"|"closing", messages?:[], opponentLastMessage?:{...} }
  - Resp: { success, data.{ content, avatarId, timestamp } }

### TTS
- POST /tts/synthesize
  - Body: { text: string, persona: Persona }
  - Behavior: uses ElevenLabs if ELEVEN_API_KEY set; otherwise mock (silent WAV)
  - Resp: { audio_path, duration }

### Rendering & Clips
- POST /episodes/{id}/render → runs md_podcast_build.py; returns assets
- POST /episodes/{id}/clips → { n, minSec, maxSec, filterSpeaker?, font?, bg?, wf_ratio? }
- GET /episodes/{id}/clips → { episode_id, clips[] }

### TTS Voice Config
- GET /config/tts-voices → contents of config/tts_voices.yaml or { voices: {} }
- PUT /config/tts-voices → write config (YAML)

### Evidence utilities (demo)
- GET /episodes/{id}/evidence
- GET /episodes/{id}/turns/{turn_id}/evidence
- GET /episodes/{id}/sources
- GET /citations/{citation_id}

### Notes
- Authentication: none in dev; add API key headers for prod.
- Long-running: render/clips are synchronous in MVP; can move to background tasks later.
