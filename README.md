# gta-ai-labelling: GenAI Creative Asset Compliance Suite

This package provides utilities to support regulatory compliance
for GenAI-generated assets:

- **AI Watermark Embedding Tool**: Injects Google-UX/YouTube compliant visual
  disclosure watermarks onto photos and videos.


## 📖 Usage & Examples


### Embed visual AI Watermark

Run the embedder tool on an individual photo or video file:


```sh
# Apply watermark to a single photo
python watermark_embedder_main.py \
  --input path/to/your/input_image.jpg \
  --output_dir ./demo_output

# Apply watermark to a single video
python watermark_embedder_main.py \
  --input path/to/your/input_video.mp4 \
  --output_dir ./demo_output
```

Options:

- `-i` / `--input`: (Required) Path to input photo or video file.
- `-o` / `--output_dir`: Directory to store the output watermarked files.
- `-p` / `--position`: Placement corner (`top-left`, `top-right`, `bottom-left`,
  `bottom-right`).
- `-s` / `--scale`: (Optional) Manual scale factor for watermark badge.
- `-m` / `--margin`: (Optional) Margin in pixels from canvas edge (default: 12).

### Install & Use as a Python Library

Install directly from GitHub with `pip`:

```sh
pip install git+https://github.com/google-marketing-solutions/gta-ai-labelling.git
```

Embed an AI watermark onto an image or video in Python:

```python
import watermark_embedder

# Watermark an image file
watermark_embedder.embed_watermark_image(
    image_input="path/to/your/input_image.jpg",
    output_path="path/to/your/output_image.jpg",
    position="top-left",
)

# Watermark a video file (requires ffmpeg installed on PATH)
watermark_embedder.embed_watermark_video(
    input_path="path/to/your/input_video.mp4",
    output_path="path/to/your/output_video.mp4",
    position="top-left",
)
```
