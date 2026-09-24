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

"""CLI for AI Watermark Embedding Tool.

Takes a photo/video file, overlays an AI watermark per UX specs,
and saves the watermarked output file in a destination directory.

Usage:
    blaze run //corp/gtech/ads/infrastructure/gta_ai_labelling:ai_watermark_tool
        -- --input /path/to/sample.jpg --output_dir ./results_folder
"""

from collections.abc import Sequence
import os
import sys
from absl import app
from absl import flags
from absl import logging
import watermark_embedder

_INPUT = flags.DEFINE_string(
    'input',
    None,
    'Path to an input photo or video file.',
    short_name='i',
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    './watermarked_output',
    'Directory where watermarked results will be stored.',
    short_name='o',
)
_POSITION = flags.DEFINE_string(
    'position',
    'top-left',
    'Watermark position on screen: top-left (recommended), top-right, '
    'bottom-left, bottom-right.',
    short_name='p',
)
_MARGIN = flags.DEFINE_integer(
    'margin',
    12,
    'Margin in pixels from the canvas edge.',
    short_name='m',
)
_SCALE = flags.DEFINE_float(
    'scale',
    None,
    'Scale factor for watermark badge (default: automatic based on '
    'resolution).',
    short_name='s',
)

_VIDEO_EXTENSIONS = frozenset(
    {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
)


def _resolve_path(file_path: str) -> str:
  """Resolves a file path, taking into account build working directory."""
  if not file_path:
    return file_path
  if os.path.isabs(file_path):
    return file_path
  if 'BUILD_WORKING_DIRECTORY' in os.environ:
    return os.path.join(os.environ['BUILD_WORKING_DIRECTORY'], file_path)
  return os.path.abspath(file_path)


def main(argv: Sequence[str]) -> None:
  input_path = _INPUT.value
  # Check if input path passed as positional argument
  if not input_path and len(argv) > 1:
    input_path = argv[1]

  if not input_path:
    logging.error('Error: Please specify a valid file using --input.')
    sys.exit(1)

  resolved_input_path = _resolve_path(input_path)

  if not os.path.exists(resolved_input_path):
    logging.error(
        'Error: Input path "%s" does not exist. Please specify a valid file.',
        resolved_input_path,
    )
    sys.exit(1)

  if os.path.isdir(resolved_input_path):
    logging.error(
        'Error: "%s" is a directory. Only single file processing is supported.',
        resolved_input_path,
    )
    sys.exit(1)

  filename = os.path.basename(resolved_input_path)
  base_name, file_ext = os.path.splitext(filename)
  ext = file_ext.lower()
  output_filename = f'{base_name}_watermarked{file_ext}'

  resolved_output_dir = _resolve_path(_OUTPUT_DIR.value)
  os.makedirs(resolved_output_dir, exist_ok=True)
  output_path = os.path.join(resolved_output_dir, output_filename)

  logging.info('Embedding AI watermark...')
  logging.info('Input path: %s', resolved_input_path)
  logging.info('Output folder: %s', resolved_output_dir)
  logging.info('Position: %s', _POSITION.value)

  try:
    if ext in _VIDEO_EXTENSIONS:
      result = watermark_embedder.embed_watermark_video(
          input_path=resolved_input_path,
          output_path=output_path,
          position=_POSITION.value,
          margin=_MARGIN.value,
          scale=_SCALE.value,
      )
    else:
      result = watermark_embedder.embed_watermark_image(
          image_input=resolved_input_path,
          output_path=output_path,
          position=_POSITION.value,
          margin=_MARGIN.value,
          scale=_SCALE.value,
      )
    logging.info('Successfully created AI watermarked file at: %s', result)

  except (OSError, ValueError, RuntimeError) as e:
    logging.exception('Failed to process media path: %s', e)
    sys.exit(1)


if __name__ == '__main__':
  app.run(main)
