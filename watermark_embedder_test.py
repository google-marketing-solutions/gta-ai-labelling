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

import math
import os
import shutil
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import ffmpeg
from PIL import Image
from PIL import ImageChops
from PIL import ImageStat

import watermark_embedder


def _create_textured_image(width: int, height: int) -> Image.Image:
  """Creates a deterministic detailed RGB image sensitive to lossy compression."""
  noise = Image.effect_noise((width, height), 40.0)
  gradient_r = Image.linear_gradient('L').resize((width, height))
  gradient_b = gradient_r.transpose(Image.Transpose.ROTATE_90).resize(
      (width, height)
  )
  return Image.merge('RGB', (gradient_r, noise, gradient_b))


def _compute_psnr(img_a: Image.Image, img_b: Image.Image) -> float:
  """Computes PSNR in dB between two equal-sized RGB images."""
  diff = ImageChops.difference(img_a.convert('RGB'), img_b.convert('RGB'))
  rms_channels = ImageStat.Stat(diff).rms
  mse = sum(channel * channel for channel in rms_channels) / len(rms_channels)
  if mse == 0:
    return float('inf')
  return 10.0 * math.log10((255.0 * 255.0) / mse)


def _create_high_quality_video(
    output_path: str, width: int = 640, height: int = 480
) -> None:
  """Generates a real high-bitrate MP4 with video and audio streams."""
  video_in = ffmpeg.input(
      f'testsrc2=size={width}x{height}:rate=24:duration=1.5',
      f='lavfi',
  )
  audio_in = ffmpeg.input(
      'sine=frequency=440:sample_rate=48000:duration=1.5',
      f='lavfi',
  )
  stream = ffmpeg.output(
      video_in,
      audio_in,
      output_path,
      vcodec='libx264',
      crf=12,
      pix_fmt='yuv420p',
      acodec='aac',
      audio_bitrate='128k',
  )
  ffmpeg.run(stream, overwrite_output=True, quiet=True)


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

  @parameterized.named_parameters(
      dict(
          testcase_name='high_quality_jpeg',
          filename='detailed.jpeg',
          save_kwargs={'format': 'JPEG', 'quality': 95, 'subsampling': 0},
          min_size_ratio=0.85,
          min_unwatermarked_psnr_db=40.0,
      ),
      dict(
          testcase_name='lossless_png',
          filename='detailed.png',
          save_kwargs={'format': 'PNG'},
          min_size_ratio=0.95,
          min_unwatermarked_psnr_db=float('inf'),
      ),
      dict(
          testcase_name='high_quality_webp',
          filename='detailed.webp',
          save_kwargs={'format': 'WEBP', 'quality': 95},
          min_size_ratio=0.85,
          min_unwatermarked_psnr_db=37.0,
      ),
  )
  def test_embed_watermark_image_preserves_resolution_size_and_quality(
      self,
      filename,
      save_kwargs,
      min_size_ratio,
      min_unwatermarked_psnr_db,
  ):
    input_path = os.path.join(self.test_dir, filename)
    output_path = os.path.join(self.test_dir, f'watermarked_{filename}')
    source_img = _create_textured_image(640, 480)
    source_img.save(input_path, **save_kwargs)

    with Image.open(input_path) as encoded_input:
      encoded_input.load()
      expected_resolution = encoded_input.size
      input_bottom_half = encoded_input.crop((0, 240, 640, 480)).copy()

    input_size_bytes = os.path.getsize(input_path)

    watermark_embedder.embed_watermark_image(
        image_input=input_path,
        output_path=output_path,
        position='top-left',
    )

    output_size_bytes = os.path.getsize(output_path)
    with Image.open(output_path) as out_img:
      out_img.load()
      self.assertEqual(out_img.size, expected_resolution)
      output_bottom_half = out_img.crop((0, 240, 640, 480)).copy()

    psnr_db = _compute_psnr(input_bottom_half, output_bottom_half)
    with self.subTest('file_size_preserved'):
      self.assertGreaterEqual(
          output_size_bytes,
          input_size_bytes * min_size_ratio,
          f'Output size dropped from {input_size_bytes} to {output_size_bytes} '
          f'bytes ({output_size_bytes / input_size_bytes:.1%} of input).',
      )
    with self.subTest('unwatermarked_region_quality_preserved'):
      self.assertGreaterEqual(
          psnr_db,
          min_unwatermarked_psnr_db,
          f'Unwatermarked region PSNR dropped to {psnr_db:.2f} dB.',
      )

  def test_embed_watermark_video_preserves_resolution_size_and_quality(self):
    input_path = os.path.join(self.test_dir, 'high_quality_input.mp4')
    output_path = os.path.join(self.test_dir, 'high_quality_output.mp4')
    _create_high_quality_video(input_path, width=640, height=480)

    input_probe = ffmpeg.probe(input_path)
    input_video_stream = next(
        s for s in input_probe['streams'] if s['codec_type'] == 'video'
    )
    input_audio_stream = next(
        s for s in input_probe['streams'] if s['codec_type'] == 'audio'
    )
    input_size_bytes = os.path.getsize(input_path)
    input_video_bitrate = int(input_video_stream['bit_rate'])

    watermark_embedder.embed_watermark_video(
        input_path=input_path,
        output_path=output_path,
        position='top-left',
    )

    output_probe = ffmpeg.probe(output_path)
    output_video_stream = next(
        s for s in output_probe['streams'] if s['codec_type'] == 'video'
    )
    output_audio_stream = next(
        s for s in output_probe['streams'] if s['codec_type'] == 'audio'
    )
    output_size_bytes = os.path.getsize(output_path)
    output_video_bitrate = int(output_video_stream['bit_rate'])

    with self.subTest('resolution_preserved'):
      self.assertEqual(
          (
              int(output_video_stream['width']),
              int(output_video_stream['height']),
          ),
          (int(input_video_stream['width']), int(input_video_stream['height'])),
      )
    with self.subTest('frame_rate_and_audio_preserved'):
      self.assertEqual(
          output_video_stream['avg_frame_rate'],
          input_video_stream['avg_frame_rate'],
      )
      self.assertEqual(
          output_audio_stream['codec_name'], input_audio_stream['codec_name']
      )
      self.assertEqual(
          output_audio_stream['bit_rate'], input_audio_stream['bit_rate']
      )
    with self.subTest('file_size_preserved'):
      self.assertGreaterEqual(
          output_size_bytes,
          input_size_bytes * 0.80,
          f'Output video size dropped from {input_size_bytes} to '
          f'{output_size_bytes} bytes '
          f'({output_size_bytes / input_size_bytes:.1%} of input).',
      )
    with self.subTest('video_bitrate_preserved'):
      self.assertGreaterEqual(
          output_video_bitrate,
          input_video_bitrate * 0.80,
          f'Output video bitrate dropped from {input_video_bitrate} to '
          f'{output_video_bitrate} bps '
          f'({output_video_bitrate / input_video_bitrate:.1%} of input).',
      )

  @parameterized.named_parameters(
      ('jpeg', 'meta.jpeg', {'format': 'JPEG', 'quality': 95}),
      ('png', 'meta.png', {'format': 'PNG'}),
      ('webp', 'meta.webp', {'format': 'WEBP', 'quality': 95}),
  )
  def test_embed_watermark_image_preserves_metadata(
      self, filename, save_kwargs
  ):
    input_path = os.path.join(self.test_dir, filename)
    output_path = os.path.join(self.test_dir, f'out_{filename}')
    img = Image.new('RGB', (320, 240), color=(40, 80, 120))
    exif = Image.Exif()
    exif[0x010E] = 'AI Watermark Test Asset'
    fake_icc = b'fake_icc_profile_bytes_1234'
    img.save(
        input_path, exif=exif.tobytes(), icc_profile=fake_icc, **save_kwargs
    )

    watermark_embedder.embed_watermark_image(
        image_input=input_path,
        output_path=output_path,
    )

    with Image.open(output_path) as out_img:
      self.assertEqual(out_img.info.get('icc_profile'), fake_icc)
      self.assertEqual(out_img.getexif().get(0x010E), 'AI Watermark Test Asset')

  def test_embed_watermark_image_in_memory_preserves_metadata(self):
    img = Image.new('RGB', (320, 240), color=(40, 80, 120))
    exif = Image.Exif()
    exif[0x010E] = 'In-Memory Asset'
    img.info['exif'] = exif.tobytes()
    img.info['icc_profile'] = b'fake_icc_bytes'
    img.info['dpi'] = (300, 300)
    img.info['xmp'] = b'<xmp>test</xmp>'

    result_img = watermark_embedder.embed_watermark_image(image_input=img)

    self.assertEqual(result_img.info.get('exif'), img.info['exif'])
    self.assertEqual(result_img.info.get('icc_profile'), b'fake_icc_bytes')
    self.assertEqual(result_img.info.get('dpi'), (300, 300))
    self.assertEqual(result_img.info.get('xmp'), b'<xmp>test</xmp>')

  @parameterized.named_parameters(
      dict(
          testcase_name='bt709_with_unknown_range',
          stream_color_fields={
              'color_space': 'bt709',
              'color_primaries': 'bt709',
              'color_transfer': 'bt709',
              'color_range': 'unknown',
          },
          expected_pairs={
              '-colorspace': 'bt709',
              '-color_primaries': 'bt709',
              '-color_trc': 'bt709',
          },
          omitted_flags=('-color_range',),
      ),
      dict(
          testcase_name='rgb_gbr_colorspace_omitted_for_yuv420p',
          stream_color_fields={
              'color_space': 'gbr',
              'color_primaries': 'bt709',
              'color_transfer': 'iec61966-2-1',
              'color_range': 'pc',
          },
          expected_pairs={
              '-color_primaries': 'bt709',
              '-color_trc': 'iec61966-2-1',
              '-color_range': 'pc',
          },
          omitted_flags=('-colorspace',),
      ),
  )
  def test_embed_watermark_video_passes_quality_and_color_metadata_flags(
      self, stream_color_fields, expected_pairs, omitted_flags
  ):
    mock_probe = self.enter_context(
        mock.patch.object(ffmpeg, 'probe', autospec=True, spec_set=True)
    )
    mock_run = self.enter_context(
        mock.patch.object(ffmpeg, 'run', autospec=True, spec_set=True)
    )
    mock_probe.return_value = {
        'streams': [{
            'codec_type': 'video',
            'width': 1920,
            'height': 1080,
            **stream_color_fields,
        }]
    }
    mock_run.side_effect = _fake_ffmpeg_run

    input_video_path = os.path.join(self.test_dir, 'hd_input.mp4')
    with open(input_video_path, 'wb') as f:
      f.write(b'dummy_video_contents')

    output_file = os.path.join(self.test_dir, 'hd_output.mp4')
    watermark_embedder.embed_watermark_video(
        input_path=input_video_path,
        output_path=output_file,
    )

    compiled_args = ffmpeg.get_args(
        mock_run.call_args.args[0], overwrite_output=True
    )
    arg_pairs = {
        compiled_args[i]: compiled_args[i + 1]
        for i in range(len(compiled_args) - 1)
        if compiled_args[i].startswith('-')
    }
    self.assertEqual(arg_pairs.get('-crf'), '16')
    self.assertEqual(arg_pairs.get('-movflags'), '+faststart')
    for flag, expected_value in expected_pairs.items():
      self.assertEqual(arg_pairs.get(flag), expected_value)
    for flag in omitted_flags:
      self.assertNotIn(flag, compiled_args)


if __name__ == '__main__':
  absltest.main()
