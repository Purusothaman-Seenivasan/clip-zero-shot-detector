# clip-zero-shot-detector
Detect objects from text prompts using OpenAI’s CLIP (ViT-B/32).   Built for fast inference: dynamic quantization + parallel patch processing.

# CLIP Zero-Shot Object Detector (ViT-B/32)

Detect objects from **text prompts** using OpenAI’s CLIP (ViT-B/32).  
Built for fast inference: dynamic quantization + parallel patch processing.

---

## Highlights

- Zero-shot, promptable detection (“sharp object”, “office tool”, etc.)
- Optimized inference: ~**5.8 s → 0.56 s per image** (~10× faster) via dynamic quantization & parallel tile processing.

---

## Quickstart

Clone the repo & run the notebook:
```bash
pip install -r requirements.txt

## Run
python infer.py \
  --image pen.jpeg \
  --prompt "plant" \
  --patch-size 124 \
  --threshold 0.6 
