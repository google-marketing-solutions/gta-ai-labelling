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

"""Tests for AI Watermark Embedding Library."""

import os
import shutil
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ffmpeg
from PIL import Image

import watermark_embedder


def _fake_ffmpeg_run(
    stream_spec: Any,
    overwrite_output: bool = False,
    **_kwargs: Any,
) -> tuple[bytes, bytes]:
  args = ffmpeg.get_args(stream_spec, overwrite_output=overwrite_output)
  out_path = args[-2] if overwrite_output and args[-1] == '-y' else args[-1]
  with open(out_path, 'wb') as f:
    f.write(b'fake_video_contents')
  return b'', b''


class WatermarkEmbedderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)
    super().tearDown()

  def test_create_ai_watermark_badge(self):
    badge = watermark_embedder._create_ai_watermark_badge(scale=1.0)
    self.assertEqual(badge.mode, 'RGBA')
    self.assertGreater(badge.width, 0)
    self.assertGreater(badge.height, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name='top_left',
          position='top-left',
          expected=(10, 10),
      ),
      dict(
          testcase_name='top_right',
          position='top-right',
          expected=(940, 10),
      ),
      dict(
          testcase_name='bottom_left',
          position='bottom-left',
          expected=(10, 470),
      ),
      dict(
          testcase_name='bottom_right',
          position='bottom-right',
          expected=(940, 470),
      ),
  )
  def test_calculate_watermark_position(self, position, expected):
    x, y = watermark_embedder._calculate_watermark_position(
        canvas_w=1000,
        canvas_h=500,
        badge_w=50,
        badge_h=20,
        position=position,
        margin=10,
    )
    self.assertEqual((x, y), expected)

  @parameterized.named_parameters(
      ('rgb', 'RGB'),
      ('rgba', 'RGBA'),
      ('l', 'L'),
      ('p', 'P'),
      ('1', '1'),
  )
  def test_embed_watermark_image_pil_instance(self, mode):
    img = Image.new(mode, (640, 480))
    watermarked = watermark_embedder.embed_watermark_image(img)
    self.assertIsInstance(watermarked, Image.Image)
    self.assertEqual(watermarked.size, (640, 480))
    self.assertEqual(watermarked.mode, mode)

  @parameterized.named_parameters(
      ('png', 'sample.png', 'output.png'),
      ('jpg', 'sample.jpg', 'output.jpg'),
      ('webp', 'sample.webp', 'output.webp'),
  )
  def test_embed_watermark_image(self, input_name, output_name):
    input_image_path = os.path.join(self.test_dir, input_name)
    img = Image.new('RGB', (640, 480), color=(100, 150, 200))
    img.save(input_image_path)

    output_path = os.path.join(self.test_dir, output_name)
    output_file = watermark_embedder.embed_watermark_image(
        image_input=input_image_path,
        output_path=output_path,
        position='top-left',
    )

    self.assertEqual(output_file, output_path)
    self.assertTrue(os.path.exists(output_file))
    self.assertGreater(os.path.getsize(output_file), 0)

    with Image.open(output_file) as out_img:
      self.assertEqual(out_img.size, (640, 480))

  def test_embed_watermark_image_nonexistent_file_raises_error(self):
    nonexistent = os.path.join(self.test_dir, 'does_not_exist.png')
    output_file = os.path.join(self.test_dir, 'output.png')
    with self.assertRaises(FileNotFoundError):
      watermark_embedder.embed_watermark_image(
          nonexistent, output_path=output_file
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='with_audio',
          streams=[
              {'codec_type': 'video', 'width': 640, 'height': 480},
              {'codec_type': 'audio'},
          ],
          expect_audio_copy=True,
      ),
      dict(
          testcase_name='video_only',
          streams=[
              {'codec_type': 'video', 'width': 640, 'height': 480},
          ],
          expect_audio_copy=False,
      ),
  )
  def test_embed_watermark_video(self, streams, expect_audio_copy):
    mock_probe = self.enter_context(
        mock.patch.object(ffmpeg, 'probe', autospec=True, spec_set=True)
    )
    mock_run = self.enter_context(
        mock.patch.object(ffmpeg, 'run', autospec=True, spec_set=True)
    )
    mock_probe.return_value = {'streams': streams}
    mock_run.side_effect = _fake_ffmpeg_run

    input_video_path = os.path.join(self.test_dir, 'dummy_video.mp4')
    with open(input_video_path, 'wb') as f:
      f.write(b'dummy_video_contents')

    output_file = os.path.join(self.test_dir, 'output_video.mp4')
    result = watermark_embedder.embed_watermark_video(
        input_path=input_video_path,
        output_path=output_file,
        position='top-left',
    )

    self.assertEqual(result, output_file)
    self.assertTrue(os.path.exists(output_file))
    self.assertGreater(os.path.getsize(output_file), 0)
    mock_probe.assert_called_once_with(input_video_path)
    mock_run.assert_called_once()

    stream_arg = mock_run.call_args.args[0]
    compiled_args = ffmpeg.get_args(stream_arg, overwrite_output=True)
    self.assertIn('-filter_complex', compiled_args)
    self.assertEqual('-acodec' in compiled_args, expect_audio_copy)

  def test_embed_watermark_image_unsupported_format_raises_error(self):
    unsupported_file = os.path.join(self.test_dir, 'sample.txt')
    with open(unsupported_file, 'w') as f:
      f.write('sample content')
    output_file = os.path.join(self.test_dir, 'output.txt')
    with self.assertRaisesRegex(ValueError, 'Unsupported image file format'):
      watermark_embedder.embed_watermark_image(
          unsupported_file, output_path=output_file
      )

  def test_embed_watermark_image_rgba_to_jpeg(self):
    img = Image.new('RGBA', (640, 480), color=(100, 150, 200, 255))
    output_file = os.path.join(self.test_dir, 'output.jpg')
    result = watermark_embedder.embed_watermark_image(
        img, output_path=output_file
    )
    self.assertEqual(result, output_file)
    self.assertTrue(os.path.exists(output_file))
    with Image.open(output_file) as out_img:
      self.assertEqual(out_img.mode, 'RGB')

  def test_embed_watermark_image_unsupported_output_format_raises_error(self):
    img = Image.new('RGB', (640, 480))
    output_file = os.path.join(self.test_dir, 'output.txt')
    with self.assertRaisesRegex(ValueError, 'Unsupported output image format'):
      watermark_embedder.embed_watermark_image(
          img, output_path=output_file
      )

  def test_embed_watermark_video_nonexistent_file_raises_error(self):
    nonexistent = os.path.join(self.test_dir, 'does_not_exist.mp4')
    output_file = os.path.join(self.test_dir, 'output.mp4')
    with self.assertRaises(FileNotFoundError):
      watermark_embedder.embed_watermark_video(
          nonexistent, output_path=output_file
      )

  def test_embed_watermark_video_unsupported_input_format_raises_error(self):
    unsupported_file = os.path.join(self.test_dir, 'sample.txt')
    with open(unsupported_file, 'w') as f:
      f.write('sample content')
    output_file = os.path.join(self.test_dir, 'output.mp4')
    with self.assertRaisesRegex(ValueError, 'Unsupported video file format'):
      watermark_embedder.embed_watermark_video(
          unsupported_file, output_path=output_file
      )

  def test_embed_watermark_video_unsupported_output_format_raises_error(self):
    input_file = os.path.join(self.test_dir, 'sample.mp4')
    with open(input_file, 'wb') as f:
      f.write(b'video content')
    output_file = os.path.join(self.test_dir, 'output.txt')
    with self.assertRaisesRegex(ValueError, 'Unsupported output video format'):
      watermark_embedder.embed_watermark_video(
          input_file, output_path=output_file
      )

  def test_embed_watermark_video_probe_failure_raises_error(self):
    mock_probe = self.enter_context(
        mock.patch.object(ffmpeg, 'probe', autospec=True, spec_set=True)
    )
    mock_probe.return_value = {'streams': []}

    input_video_path = os.path.join(self.test_dir, 'dummy_video.mp4')
    with open(input_video_path, 'wb') as f:
      f.write(b'dummy_video_contents')

    output_file = os.path.join(self.test_dir, 'output_video.mp4')
    with self.assertRaisesRegex(ValueError, 'No video stream found'):
      watermark_embedder.embed_watermark_video(
          input_path=input_video_path,
          output_path=output_file,
      )

  def test_embed_watermark_video_encode_failure_raises_error(self):
    mock_probe = self.enter_context(
        mock.patch.object(ffmpeg, 'probe', autospec=True, spec_set=True)
    )
    mock_run = self.enter_context(
        mock.patch.object(ffmpeg, 'run', autospec=True, spec_set=True)
    )
    mock_probe.return_value = {
        'streams': [{'codec_type': 'video', 'width': 640, 'height': 480}]
    }
    mock_run.side_effect = ffmpeg.Error('ffmpeg', b'', b'encode failed')

    input_video_path = os.path.join(self.test_dir, 'dummy_video.mp4')
    with open(input_video_path, 'wb') as f:
      f.write(b'dummy_video_contents')

    output_file = os.path.join(self.test_dir, 'output_video.mp4')
    with self.assertRaisesRegex(ValueError, 'Unable to process video file'):
      watermark_embedder.embed_watermark_video(
          input_path=input_video_path,
          output_path=output_file,
      )


if __name__ == '__main__':
  absltest.main()
