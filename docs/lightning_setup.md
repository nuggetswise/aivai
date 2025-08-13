# Lightning.ai Setup for AI vs AI Podcast

This document provides detailed instructions for setting up and using Lightning.ai as your production environment for AI vs AI podcast generation, with a focus on GPU-accelerated processing.

## Prerequisites

- Lightning.ai account (free tier or paid)
- SSH access configured (already done with your token)
- Git repository with your AI vs AI podcast code

## Connection Details

Your Lightning.ai environment is already configured with SSH access:

```bash
ssh s_01k2hw3qwtg9e6e8hpxsydzxj3@ssh.lightning.ai
```

## Setup Steps

### 1. First-Time Setup

After connecting via SSH, set up your environment:

```bash
# Clone your repository (replace with your actual repo)
git clone https://github.com/yourusername/ai-vs-ai-podcast.git
cd ai-vs-ai-podcast

# Create necessary directories
mkdir -p data/audio data/bundles data/indices data/sources data/transcripts data/turns data/voices
mkdir -p corpus/skeptic corpus/poppulse

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env to add your API keys
nano .env
```

### 2. Configure for GPU Acceleration

Edit your `.env` file to enable GPU features:

```
# GPU Configuration
USE_GPU=true
TTS_USE_GPU=true
CUDA_VISIBLE_DEVICES=0
```

### 3. Optional: Configure Advanced Settings

For advanced GPU usage, edit `hybrid_config.toml`:

```toml
[production]
# Lightning.ai production - full features enabled
TTS_ENABLED = true
DEBUG = false
LOG_LEVEL = INFO
MAX_SEARCH_RESULTS = 12
MAX_TURNS_PER_PHASE = 4
ENABLE_CACHING = true
CORPUS_INDEXING = true
GPU_ACCELERATION = true
```

## Running Episodes on Lightning.ai

### Basic Episode

```bash
# Run a basic episode
make episode TOPIC="AI Safety and Regulation" AVATAR_A=avatars/skeptic.yaml AVATAR_B=avatars/poppulse.yaml
```

### Full Production Episode

For complete episodes with all features enabled:

```bash
# Set maximum quality
export MAX_TURNS=3
export TTS_QUALITY=high
export AUDIO_QUALITY=high

# Run production-quality episode
make episode TOPIC="The Future of Work in an AI-driven Economy" AVATAR_A=avatars/skeptic.yaml AVATAR_B=avatars/poppulse.yaml
```

## Synchronizing Between Local and Lightning.ai

### Push Local Changes to Lightning.ai

If using git:
```bash
# On local machine
git add .
git commit -m "Update for production"
git push

# On Lightning.ai
git pull
```

Using direct file transfer:
```bash
# From local machine to Lightning.ai
rsync -avz --exclude 'data/' --exclude 'node_modules/' --exclude '.git/' /Users/singhm/AIvAI/ s_01k2hw3qwtg9e6e8hpxsydzxj3@ssh.lightning.ai:~/ai-vs-ai-podcast/
```

### Pull Generated Content from Lightning.ai

```bash
# From Lightning.ai to local machine
rsync -avz s_01k2hw3qwtg9e6e8hpxsydzxj3@ssh.lightning.ai:~/ai-vs-ai-podcast/data/ /Users/singhm/AIvAI/data/
```

## Performance Optimization

To get the best performance on Lightning.ai:

1. **TTS Optimization**: Ensure GPU acceleration is enabled for Dia TTS
   ```python
   # In app/tts/dia_synth.py
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Batch Processing**: For multiple episodes, use background processing
   ```bash
   nohup make episode TOPIC="Topic 1" > episode1.log &
   nohup make episode TOPIC="Topic 2" > episode2.log &
   ```

3. **Resource Monitoring**: Check GPU usage
   ```bash
   nvidia-smi
   ```

## Troubleshooting

- **CUDA errors**: Ensure CUDA toolkit is properly installed
  ```bash
  nvcc --version
  ```

- **Memory issues**: Reduce batch sizes in TTS and embedding operations
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  export MAX_BATCH_SIZE=4
  ```

## Recommended Workflow

1. **Develop locally** - Implement features, test with minimal processing
2. **Test on Lightning.ai** - Verify features work with GPU acceleration
3. **Run production episodes** - Generate full-quality episodes with maximum settings
4. **Download outputs** - Sync generated audio and transcripts to local machine

## Reference Links

- [Lightning.ai Documentation](https://lightning.ai/docs/app/stable/)
- [PyTorch CUDA Setup](https://pytorch.org/docs/stable/notes/cuda.html)
- [Dia TTS Documentation](https://github.com/microsoft/Dia)