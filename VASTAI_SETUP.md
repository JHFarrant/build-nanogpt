# Running nanoGPT on vast.ai

## Quick Start (Easiest)

1. **Create a vast.ai instance:**
   - Go to https://www.vast.ai/
   - Search for instances with: 
     - **CUDA**: 12.1+ (for best performance)
     - **GPU Memory**: 24GB+ (RTX 4090, A100, H100 recommended)
     - **RAM**: 64GB+ (for data loading)
   - Start an instance with SSH access

2. **Connect and run:**
```bash
# SSH into instance
ssh root@<instance-ip>

# Clone and setup
git clone <your-repo-url>
cd build-nanogpt

# Run everything (install deps, download data, train)
bash run_on_vastai.sh
```

## Using Docker (Recommended for Reliability)

1. **Build Docker image:**
```bash
docker build -t nanogpt:latest .
```

2. **Run container:**
```bash
docker run --gpus all \
  --shm-size=32g \
  -v /data/models:/app/log \
  nanogpt:latest
```

## Manual Setup (if not using script)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (~100GB disk space needed)
python fineweb.py

# 3. Start training (single GPU)
python train_gpt2.py

# For multi-GPU training:
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

## Critical Considerations

### Data Storage
- **Dataset size**: ~100GB (10B tokens in tokenized form)
- **GPU memory**: Needs persistent `/app` or mounted volume for `edu_fineweb10B/` folder
- **Recommendation**: Move data to `/data` or another persistent location if training gets interrupted

### GPU Requirements
- **Minimum**: A40 (48GB) or RTX 4090 (24GB)
- **Recommended**: H100 (80GB) or A100 (80GB) for faster training
- **Training time**: ~1 hour on A100/H100 for one epoch

### Cost Optimization
- Download data during setup, don't repeat for interruptions
- Use `--shm-size=32g` in Docker to avoid RAM bottlenecks
- Training script saves checkpoints every 5K steps to `log/` folder
- Resume from checkpoint if interrupted

### Network
- FineWeb-Edu dataset is ~100GB, ensure good connection
- Consider using `tmux` or `screen` to prevent interruption:
```bash
tmux new -s training
bash run_on_vastai.sh
# Detach with Ctrl+B then D
```

## Monitoring Training

Watch logs in real-time:
```bash
tail -f log/log.txt
```

Check GPU usage:
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Refresh every second
```

## Resuming Interrupted Training

Checkpoints are saved every 5K steps in `log/model_*.pt`. To resume from a checkpoint:

Edit `train_gpt2.py` and add checkpoint loading before the training loop (around line 370):
```python
# Load checkpoint if exists
checkpoint_path = "log/model_19070.pt"  # Change to your latest checkpoint
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    raw_model.load_state_dict(checkpoint['model'])
    # Add step offset if resuming mid-training
```

## Troubleshooting

**Out of memory errors**: Reduce batch size `B` or sequence length `T` in `train_gpt2.py`

**Data download fails**: Ensure instance has outbound internet access

**GPU not detected**: Check CUDA setup: `python -c "import torch; print(torch.cuda.is_available())"`

**Training too slow**: Check GPU utilization with `nvidia-smi`, ensure you're using GPU
