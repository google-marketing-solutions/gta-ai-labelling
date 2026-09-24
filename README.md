# gta-ai-labelling: GenAI Creative Asset Compliance Suite

This package provides two primary utilities to support regulatory compliance
for GenAI-generated assets:

1. **AI Compliance Verification Tool**: Checks assets for digital SynthID
   watermarks and C2PA metadata presence/integrity.
2. **AI Watermark Embedding Tool**: Injects Google-UX/YouTube compliant visual
   disclosure watermarks onto photos and videos.

For full architectural specifications, design standards, and API guides, see the [AI Compliance & Watermarking Tools (Design Doc)](https://docs.google.com/document/d/1VQ2aqAze7htlbIShevF235woRShQ2s6JO14nxZE-aws/edit).

---

## 🛠️ Build Targets

Build both tools using Blaze:

```sh
# Build the Compliance Verifier CLI
blaze build //corp/gtech/ads/infrastructure/gta_ai_labelling:compliance_verifier

# Build the Watermark Embedder CLI
blaze build //corp/gtech/ads/infrastructure/gta_ai_labelling:ai_watermark_tool
```

---

## 🧪 Running Unit Tests

Run verification and embedding unit tests under the Blaze test runner:

```sh
# Run Verifier Tests
blaze test //corp/gtech/ads/infrastructure/gta_ai_labelling:compliance_verifier_test

# Run Embedder Tests
blaze test //corp/gtech/ads/infrastructure/gta_ai_labelling:watermark_embedder_test
```

---

## 📖 Usage & Examples

### 1. Verify Asset Compliance

Run the verifier directly and supply an asset file:

```sh
blaze run //corp/gtech/ads/infrastructure/gta_ai_labelling:compliance_verifier -- \
  --file path/to/your/asset.jpg
```

Options:

- `--file`: (Required) Path to the asset to verify.
- `--modality`: (Optional) Modality to verify (`text`, `image`, `video`, or
  `audio`). If omitted, the tool auto-detects from the file extension.

- `--json`: (Optional) Outputs verification results in structured JSON format
  instead of human-readable text.

### 2. Embed visual AI Watermark

Run the embedder tool on an individual photo or video file:

```sh
# Apply watermark to a single photo
blaze run //corp/gtech/ads/infrastructure/gta_ai_labelling:ai_watermark_tool -- \
  --input path/to/your/input_image.jpg \
  --output_dir ./demo_output

# Apply watermark to a single video
blaze run //corp/gtech/ads/infrastructure/gta_ai_labelling:ai_watermark_tool -- \
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
