# AI vs AI Podcast - Makefile
# This Makefile provides targets for common operations

# Default Python interpreter
PYTHON = python
# Poetry command
POETRY = poetry
# Default port for API server
PORT = 8000
# Default debate topic
TOPIC = "AI Safety and Regulation"
# Default avatars
AVATAR_A = avatars/alex.yaml
AVATAR_B = avatars/nova.yaml
# Max turns per phase
MAX_TURNS = 2

.PHONY: help install update run dev seed seed-avatar test clean frontend build-frontend local-dev lightning-setup lightning-deploy sync-to-lightning

# Default target
help:
	@echo "AI vs AI Podcast Makefile - Hybrid Development"
	@echo ""
	@echo "LOCAL DEVELOPMENT:"
	@echo "  local-dev    - Start full local stack (frontend + backend)"
	@echo "  install      - Install dependencies using Poetry and frontend (npm or pnpm)"
	@echo "  run          - Run the API server"
	@echo "  dev          - Run the API server in development mode (auto-reload)"
	@echo "  frontend     - Run the Next.js frontend (npm or pnpm)"
	@echo ""
	@echo "LIGHTNING.AI PRODUCTION:"
	@echo "  lightning-setup   - Set up project on Lightning.ai"
	@echo "  lightning-deploy  - Deploy and run full episode on Lightning.ai"
	@echo "  sync-to-lightning - Sync local changes to Lightning.ai"
	@echo ""
	@echo "GENERAL:"
	@echo "  seed         - Seed the index with all avatar corpus data"
	@echo "  seed-avatar  - Seed the index with a specific avatar"
	@echo "  episode      - Run a debate episode (TOPIC=\"Your Topic\" AVATAR_A=path AVATAR_B=path)"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean temporary files and caches"
	@echo ""
	@echo "Environment setup:"
	@echo "  cp .env.example .env  # Then edit .env with your API keys"

# Local development - start both frontend and backend
local-dev:
	@echo "🚀 Starting local development environment..."
	@echo "Backend will run on http://localhost:8000"
	@echo "Frontend will run on http://localhost:3000"
	@echo ""
	@echo "Starting backend in background..."
	$(POETRY) run uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT) &
	@sleep 3
	@echo "Starting frontend..."
	cd frontend && (npm run dev || pnpm run dev)

# Install dependencies using Poetry
install:
	$(POETRY) install
	@echo "Installing dependencies for frontend..."
	cd frontend && npm install || pnpm install

# Update dependencies
update:
	$(POETRY) update
	cd frontend && npm update || pnpm update

# Run API server
run:
	$(POETRY) run uvicorn app.main:app --host 0.0.0.0 --port $(PORT)

# Run API server in development mode (auto-reload)
dev:
	$(POETRY) run uvicorn app.main:app --reload --host 0.0.0.0 --port $(PORT)

# Seed index with avatar corpus
seed:
	@echo "Seeding index with all avatar corpus data..."
	@for avatar in avatars/*.yaml; do \
		avatar_name=$$(basename $$avatar .yaml); \
		if [ -f "corpus/$$avatar_name/links.csv" ]; then \
			echo "Seeding $$avatar_name..."; \
			$(POETRY) run $(PYTHON) scripts/seed_index.py --avatar $$avatar --sources corpus/$$avatar_name/links.csv; \
		else \
			echo "Warning: No links.csv found for $$avatar_name"; \
		fi; \
	done

# Seed index with specific avatar corpus
seed-avatar:
	@if [ -z "$(AVATAR)" ]; then \
		echo "Error: AVATAR parameter is required"; \
		echo "Usage: make seed-avatar AVATAR=path/to/avatar.yaml SOURCES=path/to/links.csv"; \
		exit 1; \
	fi
	@if [ -z "$(SOURCES)" ]; then \
		echo "Error: SOURCES parameter is required"; \
		echo "Usage: make seed-avatar AVATAR=path/to/avatar.yaml SOURCES=path/to/links.csv"; \
		exit 1; \
	fi
	@echo "Seeding index with $(AVATAR) corpus..."
	$(POETRY) run $(PYTHON) scripts/seed_index.py --avatar $(AVATAR) --sources $(SOURCES)

# Run debate episode
episode:
	@echo "Running debate episode on topic: $(TOPIC)"
	@echo "Avatar A: $(AVATAR_A)"
	@echo "Avatar B: $(AVATAR_B)"
	$(POETRY) run $(PYTHON) scripts/run_episode.py --topic "$(TOPIC)" --avatar-a $(AVATAR_A) --avatar-b $(AVATAR_B) --turns $(MAX_TURNS)

# Run tests
test:
	$(POETRY) run pytest

# Clean temporary files and caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

# Run the Next.js frontend
frontend:
	cd frontend && npm run dev || pnpm run dev

# Build the Next.js frontend
build-frontend:
	cd frontend && npm run build || pnpm run build

# Lightning.ai setup
lightning-setup:
	@echo "🌩️ Setting up project on Lightning.ai..."
	@echo "Run this command to connect:"
	@echo "ssh s_01k2hw3qwtg9e6e8hpxsydzxj3@ssh.lightning.ai"
	@echo ""
	@echo "For complete setup instructions, see: docs/lightning_setup.md"
	@echo ""
	@echo "Quick setup:"
	@echo "git clone <your-repo-url> ai-vs-ai-podcast"
	@echo "cd ai-vs-ai-podcast"
	@echo "make install"
	@echo "cp .env.example .env"
	@echo "# Edit .env with your API keys"

# Deploy to Lightning.ai for full episode generation
lightning-deploy:
	@echo "🌩️ Deploying to Lightning.ai for full episode generation..."
	@echo "This will run the complete pipeline with GPU acceleration"
	@echo "SSH into Lightning.ai and run:"
	@echo "cd ai-vs-ai-podcast"
	@echo "make episode TOPIC=\"$(TOPIC)\" AVATAR_A=$(AVATAR_A) AVATAR_B=$(AVATAR_B)"

# Sync local changes to Lightning.ai
sync-to-lightning:
	@echo "📡 Syncing local changes to Lightning.ai..."
	@echo "Upload your latest changes to Lightning.ai:"
	@echo "1. Push to git: git add . && git commit -m 'Local changes' && git push"
	@echo "2. On Lightning.ai: git pull"
	@echo "3. Or use rsync/scp to sync specific files"

# Create necessary directories (called by other targets as needed)
create-dirs:
	mkdir -p data/audio data/bundles data/indices data/sources data/transcripts data/turns data/voices
	mkdir -p corpus
