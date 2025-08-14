## Podcast Prompt System Improvement Checklist

Use this checklist to upgrade prompts, personas, and orchestration for more natural, topic-adaptive episodes. Check items as you complete them.

### Personas and Voices
- [x] Assign distinct `voice.speaker_tag`s and align format with Dia TTS expectations
  - [x] Update `avatars/alex.yaml` → `voice.speaker_tag: "[S1]"`
  - [x] Update `avatars/nova.yaml` → `voice.speaker_tag: "[S2]"`
- [ ] Tune `voice.speed`, `voice.pitch`, `voice.emotion` to complementary settings (avoid both sounding identical)
- [x] Add a `style_toolkit` to each avatar YAML with:
  - [x] `contractions: true` (enable natural phrasing)
  - [x] `sentence_length: "varied"` (mix short/long)
  - [x] `rhetorical_devices` (e.g., contrast pairs, definition→example)
  - [x] `banter_moves` (e.g., ask follow-up question after counterpoint)
- [ ] Add short persona-specific “do/don’t” lists (humor level, jargon, disfluency tolerance)

### Unified Output Schema (spoken text clean; references separate)
- [ ] Adopt a JSON-like output schema for all speaking prompts (producer/LLM side):
  - [ ] `turn_id`, `speaker`, `topic`
  - [ ] `beats[]` each with: `label`, `target_duration_s`, `text` (spoken; no citations), `emotion`, `prosody { pace, pitch, pauses }`
  - [ ] `references[]` containing citation IDs that support specific claims (e.g., `S#`, `L#`)
- [ ] Spoken `text` must NOT include `[S#]`/`[L#]` inline; keep citations in `references`
- [ ] Orchestrator and TTS use `beats[].text` only for audio

### Commentator Prompt (naturalness + interaction)
- [x] Encourage contractions, short sentences, and varied rhythm
- [x] Allow light, persona-appropriate disfluencies (sparingly)
- [x] Add “interaction triggers”: end 1–2 beats with a direct question/hand-off
- [x] Define a small library of opening/mid/closing hooks and rotate per episode to avoid repetition [[memory:6133904]]
- [x] Provide 1–2 few-shot examples per persona using the schema (factual beat, banter beat, uncertainty beat)

### Style Prompt (voice polish without altering facts)
- [x] Enforce per-beat prosody: set/adjust `pace`, `pitch`, `pauses`, `target_duration_s`
- [x] Maintain strict preservation of claims and references (no factual edits)
- [x] Add a “variation policy” to rotate opening/transition/closing patterns across episodes [[memory:6133904]]
- [x] Provide a persona “style toolkit” section and instruct selection/rotation (avoid repeating last episode’s choice)

### Verifier Prompt (fact integrity with natural delivery)
- [x] Validate support in `references` rather than inline citations
- [x] If a beat’s claim lacks support:
  - [x] Soften to persona `default_unknown`, or
  - [x] Move it to a non-factual/banter beat without references
- [x] Preserve persona voice and emotional annotations

### Researcher Prompt (topic adaptability)
- [x] Add a “topic mini-brief” output:
  - [x] Key terms/definitions, top 3 angles, risky ambiguities
- [x] Normalize references with stable IDs (`S1..`, `L1..`), include source metadata
- [x] Respect corpus manifes ts (Alex academic/historical; Nova last-30-days/trending)
- [x] Emit contradictions, omissions, and confidence levels

### Episode Orchestration (run-of-show)
- [ ] Introduce run-of-show variables: `topic`, `audience`, `tone_scale(0–10)`, `time_budget_s`, `beats_plan[]`
- [ ] Provide beat libraries to mix and match:
  - [ ] Define terms → Example → Evidence-backed claim → Counterpoint → Synthesis → Audience takeaway
- [ ] Add “topic pivots” for weak evidence scenes (uncertainty framing, analogy, implications)
- [ ] Add turn-taking rules (handoffs, callbacks, direct questions)
- [x] Enforce non-mirroring intros/outros: distinct hook buckets per persona; regenerate on high similarity

### Corpus and Retrieval Alignment
- [ ] Ensure retrieval respects each avatar’s `corpus/*/manifest.yaml` weights and grounding rules
- [ ] Add a retrieval config sanity check (warn if time window/weights not met)
- [ ] Keep manifests current (`last_updated`, `source_count`)

### Quality Assurance
- [ ] Few-shot exemplars per prompt and persona (multi-domain to improve transfer)
- [ ] Prompt schema validator (CI step) to catch format drift
- [ ] Audio review checklist: natural pacing, varied hooks, clear handoffs, no citations in speech

### Implementation Steps (suggested order)
- [ ] Update `avatars/alex.yaml` + `avatars/nova.yaml` (`speaker_tag`s; add `style_toolkit`)
- [ ] Refactor `app/prompts/commentator.txt`, `style.txt`, `verifier.txt`, `researcher.txt` to the unified schema
- [ ] Add few-shot exemplars to each prompt (factual, banter, uncertainty)
- [ ] Update orchestrator to consume `beats[]` and `references[]`; send only `text` to TTS
- [ ] Add rotation logic for openings/transitions/closings to enforce per-episode variation [[memory:6133904]]
- [ ] Add Makefile targets (e.g., `make demo`) to generate a short sample episode for QA

### Optional Enhancements
- [ ] Add dynamic `audience` parameter (general, policy, developer) to adjust jargon and examples
- [ ] Introduce empathy/energy sliders per persona that map to prosody defaults
- [ ] Add cross-persona callbacks (e.g., “As Alex noted…”) when evidence aligns

Notes:
- Keep spoken output clean and conversational; place all citations in metadata.
- Avoid hardcoding phrases; rotate hooks and closers automatically per episode [[memory:6133904]].

