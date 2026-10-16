# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AI Watermark Embedding Library.

Adds an AI watermark label to photos and videos based on Google UX specs.
"""

import glob
import os
import tempfile
from typing import Any, Optional, Union

from absl import logging
import cv2
import ffmpeg
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Supported file extensions
_IMAGE_EXTENSIONS = frozenset(
    {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}
)
_VIDEO_EXTENSIONS = frozenset(
    {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
)


def _get_watermark_font(font_size: int = 12) -> Any:
  """Loads Roboto Medium font for the AI watermark badge."""
  font_names = [
      'Roboto-Medium.ttf',
      'LiberationSans-Bold.ttf',
      'LiberationSans-Regular.ttf',
      'DejaVuSans-Bold.ttf',
  ]

  for font_name in font_names:
    try:
      return ImageFont.truetype(font_name, font_size)
    except (OSError, ValueError):
      pass
  try:
    return ImageFont.load_default(size=font_size)
  except TypeError:
    return ImageFont.load_default()


def _create_ai_watermark_badge(
    text: str = 'AI',
    base_font_size: int = 12,
    scale: float = 1.0,
    text_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    bg_color: tuple[int, int, int, int] = (0, 0, 0, 153),
) -> Image.Image:
  """Creates an RGBA PIL Image containing the AI watermark badge.

  Badge Spec:
  - Icon: 12x12px circled 'i' info icon
  - Text: "AI", 12px Roboto Medium (weight 500), line-height 16px
  - Gap between icon & text: 2px
  - Background: 60% opacity black (#000, alpha 153)
  - Total size: ~26px width x 16px height (at 1.0 scale)

  Args:
    text: Text label to draw (default 'AI').
    base_font_size: Base font size in pixels (default 12).
    scale: Scaling factor for resolution adjustment.
    text_color: RGBA color tuple for text and icon.
    bg_color: RGBA color tuple for rounded pill background.

  Returns:
    PIL Image in RGBA mode with transparency.
  """
  scaled_font_size = max(10, int(base_font_size * scale))
  font = _get_watermark_font(scaled_font_size)

  # Exact spec dimensions (scaled for resolution)
  icon_size = max(10, int(12 * scale))  # 12x12px
  gap = max(1, int(2 * scale))  # 2px gap per UX spec
  padding_x = max(2, int(3 * scale))  # 3px side padding (~26px total width)
  padding_y = max(
      1, int(2 * scale)
  )  # 2px top/bottom padding (16px total height)

  # Measure text bounding box
  dummy_img = Image.new('RGBA', (1, 1))
  dummy_draw = ImageDraw.Draw(dummy_img)
  bbox = dummy_draw.textbbox((0, 0), text, font=font)
  text_w = bbox[2] - bbox[0]
  text_h = bbox[3] - bbox[1]

  content_w = icon_size + gap + text_w
  content_h = max(icon_size, text_h)

  badge_w = content_w + (padding_x * 2)
  badge_h = max(int(16 * scale), content_h + (padding_y * 2))
  corner_radius = badge_h // 2

  badge = Image.new('RGBA', (badge_w, badge_h), (0, 0, 0, 0))
  draw = ImageDraw.Draw(badge)

  # Draw background pill shape (60% opacity black)
  draw.rounded_rectangle(
      [(0, 0), (badge_w - 1, badge_h - 1)],
      radius=corner_radius,
      fill=bg_color,
  )

  # Draw info icon (12x12px circled 'i')
  icon_x = padding_x
  icon_y = (badge_h - icon_size) // 2
  icon_rect = [icon_x, icon_y, icon_x + icon_size - 1, icon_y + icon_size - 1]
  stroke_w = max(1, int(1.5 * scale))

  # Outer 12x12 circle
  draw.ellipse(icon_rect, outline=text_color, width=stroke_w)

  # 'i' inside the circle (dot + line)
  dot_r = max(1, int(1 * scale))
  dot_cx = icon_x + icon_size // 2
  dot_cy = icon_y + int(icon_size * 0.3)
  draw.ellipse(
      [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
      fill=text_color,
  )

  line_x = dot_cx
  line_y1 = icon_y + int(icon_size * 0.45)
  line_y2 = icon_y + int(icon_size * 0.75)
  draw.line(
      [(line_x, line_y1), (line_x, line_y2)],
      fill=text_color,
      width=stroke_w,
  )

  # Draw text next to icon with 2px gap
  text_x = icon_x + icon_size + gap
  text_y = (badge_h - text_h) // 2 - bbox[1]
  draw.text((text_x, text_y), text, font=font, fill=text_color)

  return badge


def _compute_scale_factor(
    width: int, height: int, manual_scale: Optional[float] = None
) -> float:
  """Computes appropriate watermark scale based on image dimensions."""
  if manual_scale is not None and manual_scale > 0:
    return manual_scale

  # Base reference resolution is 480p (480px short edge)
  min_dim = min(width, height)
  scale = max(1.0, min_dim / 480.0)
  return scale


def _calculate_watermark_position(
    canvas_w: int,
    canvas_h: int,
    badge_w: int,
    badge_h: int,
    position: str = 'top-left',
    margin: int = 12,
) -> tuple[int, int]:
  """Calculates x, y coordinates for positioning the watermark badge."""
  pos_lower = position.lower()
  if pos_lower == 'top-left':
    x = margin
    y = margin
  elif pos_lower == 'top-right':
    x = canvas_w - badge_w - margin
    y = margin
  elif pos_lower == 'bottom-left':
    x = margin
    y = canvas_h - badge_h - margin
  elif pos_lower == 'bottom-right':
    x = canvas_w - badge_w - margin
    y = canvas_h - badge_h - margin
  else:
    # Default to top-left
    x = margin
    y = margin

  # Keep within bounds
  x = max(0, min(x, canvas_w - badge_w))
  y = max(0, min(y, canvas_h - badge_h))
  return x, y


def embed_watermark_image(
    image_input: Union[str, Image.Image],
    output_path: Optional[str] = None,
    position: str = 'top-left',
    margin: int = 12,
    scale: Optional[float] = None,
) -> Union[str, Image.Image]:
  """Embeds an AI watermark badge on an image per YouTube UX specifications.

  Overlays a semi-transparent visual disclosure badge (12px Roboto Medium
  "AI" label with a circled 'i' info icon on a 60% opacity black pill
  background) onto an image file or PIL Image instance.

  Args:
    image_input: Path to the input image file (supported formats: .png, .jpg,
      .jpeg, .webp, .bmp, .tiff, .gif), or an existing PIL.Image.Image instance.
    output_path: Optional destination file path where the watermarked image will
      be saved. Required when `image_input` is a file path string. If omitted
      when `image_input` is a PIL Image, the modified in-memory Image is
      returned.
    position: Corner placement of the badge. Supported values: 'top-left'
      (default/recommended), 'top-right', 'bottom-left', 'bottom-right'.
    margin: Distance in pixels from the canvas edge to the badge. Defaults to
      12.
    scale: Optional manual scaling factor (e.g. 1.0, 1.5, 2.0). When None, the
      badge is automatically scaled based on image resolution relative to a 480p
      baseline.

  Returns:
    The file path (str) to the saved watermarked image if `output_path` is
    provided, or the watermarked PIL.Image.Image instance if `output_path` is
    None.

  Raises:
    FileNotFoundError: If `image_input` is a file path that does not exist.
    ValueError: If `image_input` or `output_path` has an unsupported file
      extension, or if `output_path` is omitted when `image_input` is a file
      path string.
  """
  if isinstance(image_input, str):
    if not os.path.exists(image_input):
      raise FileNotFoundError(f'Image file not found: {image_input}')
    ext = os.path.splitext(image_input)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
      raise ValueError(f'Unsupported image file format: {ext}')
    if not output_path:
      raise ValueError(
          'output_path is required when image_input is a file path.'
      )
    with Image.open(image_input) as img:
      return embed_watermark_image(
          img,
          output_path=output_path,
          position=position,
          margin=margin,
          scale=scale,
      )

  # Process PIL Image
  orig_mode = image_input.mode
  img_rgba = image_input.convert('RGBA')
  w, h = img_rgba.size

  calc_scale = _compute_scale_factor(w, h, scale)
  badge = _create_ai_watermark_badge(scale=calc_scale)

  margin_scaled = int(margin * calc_scale)
  x, y = _calculate_watermark_position(
      w, h, badge.width, badge.height, position=position, margin=margin_scaled
  )

  img_rgba.paste(badge, (x, y), badge)

  watermarked_img = (
      img_rgba.convert(orig_mode) if orig_mode != 'RGBA' else img_rgba
  )

  if output_path:
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
      raise ValueError(f'Unsupported output image format: {ext}')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if ext in ('.jpg', '.jpeg') and watermarked_img.mode == 'RGBA':
      watermarked_img = watermarked_img.convert('RGB')
    watermarked_img.save(output_path)
    return output_path

  return watermarked_img


def _get_video_fps(video_path: str) -> float:
  """Probes and returns the FPS (frames per second) of a video file.

  Args:
    video_path: Path to the video file.

  Returns:
    The frame rate as a float.

  Raises:
    ValueError: If unable to determine video FPS.
  """
  if hasattr(cv2, 'VideoCapture'):
    fps = None
    try:
      cap = cv2.VideoCapture(video_path)
      if cap is not None and cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
    except (OSError, ValueError, TypeError, AttributeError, cv2.error) as exc:
      logging.warning(
          'Failed to probe video FPS with OpenCV for %s: %s', video_path, exc
      )
    if fps is not None:
      if fps > 0:
        return fps
      logging.warning(
          'OpenCV VideoCapture returned non-positive FPS (%s) for %s.',
          fps,
          video_path,
      )

  data = None
  try:
    data = ffmpeg.probe(video_path)
  except (OSError, ffmpeg.Error) as exc:
    logging.warning(
        'Failed to probe video FPS with ffmpeg.probe for %s: %s',
        video_path,
        exc,
    )

  if isinstance(data, dict) and data.get('streams'):
    video_streams = [
        s for s in data['streams'] if s.get('codec_type') == 'video'
    ]
    stream = video_streams[0] if video_streams else data['streams'][0]
    rate_str = stream.get('r_frame_rate', '')
    if rate_str:
      try:
        if '/' in rate_str:
          num, den = rate_str.split('/')
          fps = float(num) / float(den)
        else:
          fps = float(rate_str)
        if fps > 0:
          return fps
      except (ValueError, ZeroDivisionError) as exc:
        logging.warning(
            'Failed to parse frame rate "%s" for %s: %s',
            rate_str,
            video_path,
            exc,
        )

  raise ValueError(f'Unable to determine FPS for video: {video_path}')


def embed_watermark_video(
    input_path: str,
    output_path: str,
    position: str = 'top-left',
    margin: int = 12,
    scale: Optional[float] = None,
) -> str:
  """Embeds an AI watermark badge on a video per YouTube UX specifications.

  Extracts video frames using FFmpeg, overlays the AI watermark badge on each
  frame, and re-encodes the final output video with original audio streams
  preserved.

  Args:
    input_path: Path to the input video file (supported formats: .mp4, .avi,
      .mov, .mkv, .webm, .flv, .wmv).
    output_path: Destination file path where the watermarked video will be
      saved.
    position: Corner placement of the badge. Supported values: 'top-left'
      (default/recommended), 'top-right', 'bottom-left', 'bottom-right'.
    margin: Distance in pixels from each frame edge to the badge. Defaults to
      12.
    scale: Optional manual scaling factor. When None, the badge is automatically
      scaled based on video resolution relative to a 480p baseline.

  Returns:
    Path to the watermarked output video file.

  Raises:
    FileNotFoundError: If `input_path` does not exist.
    ValueError: If `input_path` or `output_path` has an unsupported video file
      extension, or if FFmpeg frame extraction, watermarking, or video
      re-encoding fails.
  """
  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Video file not found: {input_path}')

  ext = os.path.splitext(input_path)[1].lower()
  if ext not in _VIDEO_EXTENSIONS:
    raise ValueError(f'Unsupported video file format: {ext}')

  out_ext = os.path.splitext(output_path)[1].lower()
  if out_ext not in _VIDEO_EXTENSIONS:
    raise ValueError(f'Unsupported output video format: {out_ext}')

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  fps = _get_video_fps(input_path)

  # Use FFmpeg frame extraction and watermarking
  with tempfile.TemporaryDirectory() as tmp_dir:
    extract_pattern = os.path.join(tmp_dir, 'frame_%06d.png')
    extract_stream = ffmpeg.output(
        ffmpeg.input(input_path),
        extract_pattern,
        vsync='0',
    )
    try:
      ffmpeg.run(extract_stream, overwrite_output=True, quiet=True)
    except (ffmpeg.Error, OSError) as exc:
      raise ValueError(
          f'Unable to process video file: {input_path}'
      ) from exc

    frame_files = sorted(glob.glob(os.path.join(tmp_dir, 'frame_*.png')))

    for frame_file in frame_files:
      try:
        embed_watermark_image(
            image_input=frame_file,
            output_path=frame_file,
            position=position,
            margin=margin,
            scale=scale,
        )
      except (OSError, ValueError) as exc:
        raise ValueError(
            f'Unable to process video file: {input_path}'
        ) from exc

    output_pattern = os.path.join(tmp_dir, 'frame_%06d.png')
    frames_input = ffmpeg.input(output_pattern, framerate=fps)
    original_input = ffmpeg.input(input_path)
    encode_stream = ffmpeg.output(
        frames_input.video,
        original_input['a?'],
        output_path,
        vcodec='libx264',
        acodec='copy',
        pix_fmt='yuv420p',
    )
    try:
      ffmpeg.run(encode_stream, overwrite_output=True, quiet=True)
    except (ffmpeg.Error, OSError) as exc:
      raise ValueError(
          f'Unable to process video file: {input_path}'
      ) from exc

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
      return output_path
    raise ValueError('Output video file is empty or missing.')
