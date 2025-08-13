# AIvAI Implementation Checklist

## **Phase 1: Foundation (Week 1)**

### 1. Complete the LLM Router and Configuration
- [ ] Update `app/config.py` with three-tier LLM system
  - [ ] Add REASONING_MODEL, REASONING_API_KEY, REASONING_BASE_URL
  - [ ] Add BASIC_MODEL, BASIC_API_KEY, BASIC_BASE_URL  
  - [ ] Add VL_MODEL, VL_API_KEY, VL_BASE_URL
- [ ] Implement LLM router in `app/deps.py`
  - [ ] `get_reasoning_llm()` function
  - [ ] `get_basic_llm()` function
  - [ ] `get_vl_llm()` function
  - [ ] HTTP session management with retry/backoff

### 2. Implement Core Data Models
- [ ] Fill in `app/models.py` with Pydantic models:
  - [ ] `Avatar` model
  - [ ] `Persona` model  
  - [ ] `Evidence` model
  - [ ] `Bundle` model
  - [ ] `Turn` model
  - [ ] `Episode` model
  - [ ] Input/output schemas matching pipeline YAML

### 3. Build the Retrieval Foundation
- [ ] Implement `app/retrieval/indexer.py`
  - [ ] Web scraping integration
  - [ ] Content cleaning
  - [ ] Text chunking
  - [ ] Embedding generation
  - [ ] FAISS storage
- [ ] Implement `app/retrieval/store.py`
  - [ ] FAISS wrapper
  - [ ] Metadata persistence (SQLite/Parquet)
  - [ ] Search and retrieval functions
- [ ] Implement `app/retrieval/rank.py`
  - [ ] Basic dense ranking
  - [ ] BM25 integration (optional)
  - [ ] Recency/trust weighting

## **Phase 2: Core Agents (Week 2)**

### 4. Implement the Agent Chain
- [ ] **Researcher** (`app/agents/researcher.py`)
  - [ ] Tavily search integration
  - [ ] Local corpus retrieval
  - [ ] Evidence bundle generation
  - [ ] Return format: `{claims: [], contradictions: [], omissions: []}`
- [ ] **Commentator** (`app/agents/commentator.py`)
  - [ ] Persona-locked response generation
  - [ ] Evidence bundle consumption
  - [ ] Citation management
  - [ ] Use reasoning-tier LLM
- [ ] **Verifier** (`app/agents/verifier.py`)
  - [ ] Factual claim validation
  - [ ] Citation checking
  - [ ] Unsupported statement removal/repair
  - [ ] Use basic-tier LLM
- [ ] **Style** (`app/agents/style.py`)
  - [ ] Tone adjustment without fact changes
  - [ ] Persona quirk application
  - [ ] Citation preservation
  - [ ] Use basic-tier LLM

### 5. Fill Prompt Templates
- [ ] Complete `app/prompts/researcher.txt`
- [ ] Complete `app/prompts/commentator.txt`
- [ ] Complete `app/prompts/verifier.txt`
- [ ] Complete `app/prompts/persona_template.yaml`
- [ ] Match system prompts from pipeline YAML

## **Phase 3: Orchestration (Week 3)**

### 6. Build Episode Runner
- [ ] Implement `app/orchestrator/episode_runner.py`
  - [ ] Debate rundown execution (opening → positions → crossfire → closing)
  - [ ] Agent sequence calling per turn
  - [ ] State management between turns
  - [ ] Opponent summary passing
  - [ ] Dialog history tracking
- [ ] Implement `app/orchestrator/rundown.py`
  - [ ] Phase sequencing logic
  - [ ] Turn order management
- [ ] Implement `app/orchestrator/guards.py`
  - [ ] Anti-repetition checks
  - [ ] Persona reminders
  - [ ] Fallback logic
- [ ] Implement `app/orchestrator/similarity.py`
  - [ ] Turn similarity detection
  - [ ] Redundancy prevention

### 7. Wire TTS Integration
- [ ] Complete `app/tts/dia_synth.py`
  - [ ] Text-to-speech conversion
  - [ ] Persona voice settings
  - [ ] Audio file output
- [ ] Complete `app/tts/adapter.py`
  - [ ] Multi-vendor TTS support
  - [ ] Unified interface
  - [ ] Dia TTS integration
  - [ ] Future vendor extensibility

## **Phase 4: Integration (Week 4)**

### 8. API Implementation
- [ ] Update `app/main.py` with FastAPI routes
  - [ ] `POST /api/episodes/start` → streaming SSE
  - [ ] `POST /api/research` → evidence bundles
  - [ ] `POST /api/avatars` → persona management
  - [ ] CORS configuration for frontend
- [ ] Implement streaming SSE for episode progress
- [ ] Add async generators for real-time updates

### 9. Test with Sample Data
- [ ] Create sample persona YAML files
  - [ ] `avatars/alex.yaml` (tech ethicist)
  - [ ] `avatars/nova.yaml` (AI researcher)
- [ ] Implement `scripts/run_episode.py`
- [ ] Test full episode generation
- [ ] Validate output artifacts (audio, transcript, notes)

### 10. Frontend Integration
- [ ] Connect Next.js frontend to streaming API
- [ ] Display real-time debate progress
- [ ] Show citations inline with turns
- [ ] Audio playback controls

## **Phase 5: Polish (Week 5)**

### 11. Add Advanced Features
- [ ] Source ranking with trust scores
- [ ] Domain whitelist/blacklist enforcement
- [ ] Anti-repetition guards enhancement
- [ ] Similarity checking refinement
- [ ] Audio mixing and post-processing
- [ ] Episode show notes generation

### 12. Testing and Documentation
- [ ] Unit tests for each agent
- [ ] Integration tests for full episodes
- [ ] API endpoint testing
- [ ] Update README with usage examples
- [ ] Add development setup guide

## **IO Utilities**
- [ ] Complete `app/io/scraper.py`
  - [ ] Browserless scraping (requests + readability)
  - [ ] Playwright integration for JS-heavy pages
- [ ] Complete `app/io/cleaner.py`
  - [ ] HTML to text conversion
  - [ ] Boilerplate removal
  - [ ] Language detection
- [ ] Complete `app/io/files.py`
  - [ ] Bundle read/write operations
  - [ ] Episode persistence
  - [ ] Temp directory management
- [ ] Complete `app/io/audio.py`
  - [ ] Audio file handling
  - [ ] Format conversion (WAV/MP3)

## **Scripts and Utilities**
- [ ] Complete `scripts/seed_index.py`
  - [ ] Corpus indexing
  - [ ] FAISS initialization
- [ ] Complete `scripts/run_episode.py`
  - [ ] Command-line episode runner
  - [ ] Parameter validation

## **Data Setup**
- [ ] Create directory structure in `data/`
- [ ] Set up voice samples in `data/voices/`
- [ ] Prepare seed corpus in `corpus/`
- [ ] Configure whitelist in `whitelist.yaml`

---

**Current Priority:** Start with Phase 1, Steps 1-2 (LLM router and data models) as they are foundational for everything else.