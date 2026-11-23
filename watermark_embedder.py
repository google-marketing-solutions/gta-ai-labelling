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

from collections.abc import Mapping
import dataclasses
import os
import tempfile
from typing import Any, Optional, Union, overload

import ffmpeg
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import JpegImagePlugin

# Supported file extensions
_IMAGE_EXTENSIONS = frozenset(
    {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}
)
_VIDEO_EXTENSIONS = frozenset(
    {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
)

# High-fidelity default encoding parameters to prevent visible quality loss
# when re-encoding after badge overlay.
_DEFAULT_IMAGE_QUALITY = 95
_DEFAULT_VIDEO_CRF = 16

_METADATA_SUPPORTED_IMAGE_EXTS = frozenset(
    {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
)
_DPI_SUPPORTED_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tiff'})
_PRESERVED_IMAGE_INFO_KEYS = ('exif', 'icc_profile', 'dpi', 'xmp')
_FASTSTART_VIDEO_EXTS = frozenset({'.mp4', '.mov'})
_INVALID_COLOR_TAGS = frozenset({'', 'unknown', 'unspecified', 'reserved'})
_NON_YUV_COLOR_SPACES = frozenset({'gbr', 'rgb'})
_VIDEO_COLOR_TAG_MAP = (
    ('color_space', 'colorspace'),
    ('color_primaries', 'color_primaries'),
    ('color_transfer', 'color_trc'),
    ('color_range', 'color_range'),
)


@dataclasses.dataclass(frozen=True)
class _VideoInfo:
  """Probed video stream metadata.

  Attributes:
    width: Video frame width in pixels.
    height: Video frame height in pixels.
    has_audio: Whether the source video contains an audio stream.
    color_tags: Tuple of validated `(ffmpeg_flag, value)` pairs for output color
      metadata.
  """

  width: int
  height: int
  has_audio: bool
  color_tags: tuple[tuple[str, str], ...] = ()


def _extract_image_save_kwargs(
    source_img: Image.Image,
    output_ext: str,
) -> Mapping[str, Any]:
  """Builds format-aware Pillow save kwargs that preserve source quality."""
  save_kwargs: dict[str, Any] = {}
  info = source_img.info or {}

  if output_ext in _METADATA_SUPPORTED_IMAGE_EXTS:
    if info.get('exif'):
      save_kwargs['exif'] = info['exif']
    if info.get('icc_profile'):
      save_kwargs['icc_profile'] = info['icc_profile']
  if output_ext in _DPI_SUPPORTED_IMAGE_EXTS and info.get('dpi'):
    save_kwargs['dpi'] = info['dpi']

  if output_ext in ('.jpg', '.jpeg'):
    if (
        isinstance(source_img, JpegImagePlugin.JpegImageFile)
        and 0 < len(source_img.quantization) < 5
    ):
      # Preserve exact source quantization tables and chroma subsampling.
      save_kwargs['qtables'] = source_img.quantization
      sampling = JpegImagePlugin.get_sampling(source_img)
      if sampling in (0, 1, 2):
        save_kwargs['subsampling'] = sampling
    else:
      save_kwargs['quality'] = _DEFAULT_IMAGE_QUALITY
      save_kwargs['subsampling'] = 0
    if info.get('progressive') or info.get('progression'):
      save_kwargs['progressive'] = True
  elif output_ext == '.webp':
    if info.get('xmp'):
      save_kwargs['xmp'] = info['xmp']
    save_kwargs['quality'] = _DEFAULT_IMAGE_QUALITY

  return save_kwargs


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
  text_w = int(bbox[2] - bbox[0])
  text_h = int(bbox[3] - bbox[1])

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


@overload
def embed_watermark_image(
    image_input: Union[str, Image.Image],
    output_path: str,
    position: str = ...,
    margin: int = ...,
    scale: Optional[float] = ...,
) -> str:
  ...


@overload
def embed_watermark_image(
    image_input: Image.Image,
    output_path: None = ...,
    position: str = ...,
    margin: int = ...,
    scale: Optional[float] = ...,
) -> Image.Image:
  ...


@overload
def embed_watermark_image(
    image_input: Union[str, Image.Image],
    output_path: Optional[str] = ...,
    position: str = ...,
    margin: int = ...,
    scale: Optional[float] = ...,
) -> Union[str, Image.Image]:
  ...


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
  source_info = image_input.info or {}
  for key in _PRESERVED_IMAGE_INFO_KEYS:
    if source_info.get(key):
      watermarked_img.info[key] = source_info[key]

  if output_path:
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
      raise ValueError(f'Unsupported output image format: {ext}')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if ext in ('.jpg', '.jpeg') and watermarked_img.mode == 'RGBA':
      watermarked_img = watermarked_img.convert('RGB')
    save_kwargs = _extract_image_save_kwargs(image_input, ext)
    watermarked_img.save(output_path, **save_kwargs)
    return output_path

  return watermarked_img


def _valid_color_tag(probe_key: str, value: Any) -> Optional[str]:
  """Returns the color tag if valid for yuv420p output, or None otherwise."""
  if not isinstance(value, str):
    return None
  normalized = value.lower()
  if normalized in _INVALID_COLOR_TAGS:
    return None
  if probe_key == 'color_space' and normalized in _NON_YUV_COLOR_SPACES:
    return None
  return value


def _get_video_info(video_path: str) -> _VideoInfo:
  """Probes video dimensions, audio presence, and color metadata.

  Args:
    video_path: Path to the video file.

  Returns:
    A `_VideoInfo` instance with dimensions, audio presence, and validated
    FFmpeg color metadata tags.

  Raises:
    ValueError: If unable to probe the video or determine valid dimensions.
  """
  try:
    probe_data = ffmpeg.probe(video_path)
  except (ffmpeg.Error, OSError) as exc:
    raise ValueError(f'Unable to probe video file: {video_path}') from exc

  streams = (
      probe_data.get('streams', []) if isinstance(probe_data, dict) else []
  )
  video_stream = next(
      (s for s in streams if s.get('codec_type') == 'video'),
      None,
  )
  if not video_stream:
    raise ValueError(f'No video stream found in: {video_path}')

  try:
    width = int(video_stream['width'])
    height = int(video_stream['height'])
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(
        f'Unable to determine dimensions for video: {video_path}'
    ) from exc

  if width <= 0 or height <= 0:
    raise ValueError(
        f'Invalid video dimensions ({width}x{height}) for: {video_path}'
    )

  has_audio = any(s.get('codec_type') == 'audio' for s in streams)
  color_tags = tuple(
      (ffmpeg_flag, tag)
      for probe_key, ffmpeg_flag in _VIDEO_COLOR_TAG_MAP
      if (tag := _valid_color_tag(probe_key, video_stream.get(probe_key)))
  )
  return _VideoInfo(
      width=width,
      height=height,
      has_audio=has_audio,
      color_tags=color_tags,
  )


def _build_video_output_kwargs(
    video_info: _VideoInfo,
    output_extension: str,
) -> Mapping[str, Any]:
  """Builds visually lossless FFmpeg encoding parameters."""
  output_kwargs: dict[str, Any] = {
      'vcodec': 'libx264',
      'crf': _DEFAULT_VIDEO_CRF,
      'pix_fmt': 'yuv420p',
  }
  if video_info.has_audio:
    output_kwargs['acodec'] = 'copy'
  if output_extension in _FASTSTART_VIDEO_EXTS:
    output_kwargs['movflags'] = '+faststart'
  output_kwargs.update(video_info.color_tags)
  return output_kwargs


def embed_watermark_video(
    input_path: str,
    output_path: str,
    position: str = 'top-left',
    margin: int = 12,
    scale: Optional[float] = None,
) -> str:
  """Embeds an AI watermark badge on a video per YouTube UX specifications.

  Generates a transparent AI watermark badge image and overlays it onto the
  video stream using FFmpeg while preserving original audio streams.

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
      extension, or if FFmpeg probing or video overlay encoding fails.
  """
  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Video file not found: {input_path}')

  extension = os.path.splitext(input_path)[1].lower()
  if extension not in _VIDEO_EXTENSIONS:
    raise ValueError(f'Unsupported video file format: {extension}')

  output_extension = os.path.splitext(output_path)[1].lower()
  if output_extension not in _VIDEO_EXTENSIONS:
    raise ValueError(f'Unsupported output video format: {output_extension}')

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  video_info = _get_video_info(input_path)

  calc_scale = _compute_scale_factor(video_info.width, video_info.height, scale)
  badge = _create_ai_watermark_badge(scale=calc_scale)
  margin_scaled = int(margin * calc_scale)
  x, y = _calculate_watermark_position(
      video_info.width,
      video_info.height,
      badge.width,
      badge.height,
      position=position,
      margin=margin_scaled,
  )

  with tempfile.TemporaryDirectory() as tmp_dir:
    badge_path = os.path.join(tmp_dir, 'watermark_badge.png')
    badge.save(badge_path, format='PNG')

    video_input = ffmpeg.input(input_path)
    badge_input = ffmpeg.input(badge_path)
    overlaid_video = video_input.video.overlay(badge_input, x=x, y=y)
    output_kwargs = _build_video_output_kwargs(video_info, output_extension)

    output_streams = (
        [overlaid_video, video_input.audio]
        if video_info.has_audio
        else [overlaid_video]
    )
    stream = ffmpeg.output(*output_streams, output_path, **output_kwargs)

    try:
      ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except (ffmpeg.Error, OSError) as exc:
      error_detail = ''
      if isinstance(exc, ffmpeg.Error) and exc.stderr:
        error_detail = (
            exc.stderr.decode('utf-8', errors='replace')
            if isinstance(exc.stderr, bytes)
            else str(exc.stderr)
        )
      raise ValueError(
          f'Unable to process video file: {input_path}. {error_detail}'.strip()
      ) from exc

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
      return output_path
    raise ValueError('Output video file is empty or missing.')
