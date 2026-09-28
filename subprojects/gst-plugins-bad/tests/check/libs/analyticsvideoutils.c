/* GStreamer
 * Copyright (C) 2026 Collabora Ltd
 * @author: Daniel Morin <daniel.morin@collabora.com>
 *
 * analyticsvideoutils.c
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#include <gst/check/gstcheck.h>
#include <gst/analytics/analytics.h>

/* gst_analytics_video_tensor_layout_from_dims() */

GST_START_TEST (test_layout_from_dims_2d_is_gray8)
{
  /* Verify a 2D tensor's dims resolve to a GRAY8 layout */
  gsize dims[] = { 4, 4 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_dims (dims,
          G_N_ELEMENTS (dims), &layout, &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_GRAY8);
  fail_unless_equals_int (layout.height, 4);
  fail_unless_equals_int (layout.width, 4);
  fail_unless_equals_int (layout.channels, 1);
  fail_if (layout.planar);
  fail_unless_equals_int (layout.batch_dim, -1);
  fail_unless_equals_int (layout.channels_dim, -1);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_3d_chw_is_planar_rgb)
{
  /* Verify a 3D CHW tensor's dims resolve to a planar RGB layout */
  gsize dims[] = { 3, 4, 8 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_dims (dims,
          G_N_ELEMENTS (dims), &layout, &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_RGBP);
  fail_unless (layout.planar);
  fail_unless_equals_int (layout.channels_dim, 0);
  fail_unless_equals_int (layout.height_dim, 1);
  fail_unless_equals_int (layout.width_dim, 2);
  fail_unless_equals_int (layout.height, 4);
  fail_unless_equals_int (layout.width, 8);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_3d_hwc_is_interleaved_rgb)
{
  /* Verify a 3D HWC tensor's dims resolve to an interleaved RGB layout */
  gsize dims[] = { 4, 8, 3 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_dims (dims,
          G_N_ELEMENTS (dims), &layout, &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_RGB);
  fail_if (layout.planar);
  fail_unless_equals_int (layout.channels_dim, 2);
  fail_unless_equals_int (layout.height_dim, 0);
  fail_unless_equals_int (layout.width_dim, 1);
  fail_unless_equals_int (layout.height, 4);
  fail_unless_equals_int (layout.width, 8);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_3d_single_channel_is_gray8)
{
  /* Verify a 3D tensor with a single-element channel dim resolves to
   * GRAY8 */
  gsize dims[] = { 1, 4, 8 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_dims (dims,
          G_N_ELEMENTS (dims), &layout, &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_GRAY8);
  fail_unless_equals_int (layout.channels, 1);
  fail_unless_equals_int (layout.channels_dim, 0);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_4d_nchw_is_planar_with_batch)
{
  /* Verify a 4D NCHW tensor's dims resolve to a planar layout with a
   * batch dim */
  gsize dims[] = { 1, 3, 4, 8 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_dims (dims,
          G_N_ELEMENTS (dims), &layout, &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_RGBP);
  fail_unless (layout.planar);
  fail_unless_equals_int (layout.batch_dim, 0);
  fail_unless_equals_int (layout.channels_dim, 1);
  fail_unless_equals_int (layout.height_dim, 2);
  fail_unless_equals_int (layout.width_dim, 3);
  fail_unless_equals_int (layout.height, 4);
  fail_unless_equals_int (layout.width, 8);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_4d_nhwc_is_interleaved_with_batch)
{
  /* Verify a 4D NHWC tensor's dims resolve to an interleaved layout
   * with a batch dim */
  gsize dims[] = { 1, 4, 8, 3 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_dims (dims,
          G_N_ELEMENTS (dims), &layout, &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_RGB);
  fail_if (layout.planar);
  fail_unless_equals_int (layout.batch_dim, 0);
  fail_unless_equals_int (layout.height_dim, 1);
  fail_unless_equals_int (layout.width_dim, 2);
  fail_unless_equals_int (layout.channels_dim, 3);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_rejects_out_of_range_dims_count)
{
  /* Verify dims counts outside [2, 4] are rejected as unsupported */
  gsize dims1[] = { 4 };
  gsize dims5[] = { 1, 2, 3, 4, 5 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_if (gst_analytics_video_tensor_layout_from_dims (dims1,
          G_N_ELEMENTS (dims1), &layout, &error));
  fail_unless (error != NULL);
  fail_unless (error->domain == GST_ANALYTICS_VIDEO_ERROR);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS);
  g_clear_error (&error);

  fail_if (gst_analytics_video_tensor_layout_from_dims (dims5,
          G_N_ELEMENTS (dims5), &layout, &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS);
  g_clear_error (&error);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_dims_rejects_unrecognized_channel_count)
{
  /* Verify dims with no recognizable channel count are rejected */
  /* Neither dimension is 1 or 3, so neither can be the channels dim. */
  gsize dims3[] = { 5, 5, 5 };
  gsize dims4[] = { 1, 5, 5, 5 };
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_if (gst_analytics_video_tensor_layout_from_dims (dims3,
          G_N_ELEMENTS (dims3), &layout, &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS);
  g_clear_error (&error);

  fail_if (gst_analytics_video_tensor_layout_from_dims (dims4,
          G_N_ELEMENTS (dims4), &layout, &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS);
  g_clear_error (&error);
}

GST_END_TEST;

/* test gst_analytics_video_tensor_layout_from_caps() */

GST_START_TEST (test_layout_from_caps_no_format_field_is_unknown)
{
  /* Verify caps with no format field resolve to an UNKNOWN layout format */
  gsize dims[] = { 4, 4, 3 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_UNKNOWN);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_resolves_declared_format)
{
  /* Verify a format declared in caps is resolved into the layout */
  gsize dims[] = { 4, 8, 3 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)RGB");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_RGB);
  fail_unless_equals_int (layout.height, 4);
  fail_unless_equals_int (layout.width, 8);
  fail_unless_equals_int (layout.channels, 3);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_resolves_declared_planar_format)
{
  /* Verify a declared planar format resolves channels/height/width dims
   * in planar order */
  gsize dims[] = { 3, 4, 8 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)RGBP");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless_equals_int (layout.format, GST_VIDEO_FORMAT_RGBP);
  fail_unless (layout.planar);
  fail_unless_equals_int (layout.channels_dim, 0);
  fail_unless_equals_int (layout.height_dim, 1);
  fail_unless_equals_int (layout.width_dim, 2);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_gray_channel_first)
{
  /* Verify a single channel declared format accepts the channel dim first */
  gsize dims_chw[] = { 1, 4, 8 };
  gsize dims_nchw[] = { 1, 1, 4, 8 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)GRAY8");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_caps (dims_chw,
          G_N_ELEMENTS (dims_chw), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_if (layout.planar);
  fail_unless_equals_int (layout.channels_dim, 0);
  fail_unless_equals_int (layout.height_dim, 1);
  fail_unless_equals_int (layout.width_dim, 2);

  fail_unless (gst_analytics_video_tensor_layout_from_caps (dims_nchw,
          G_N_ELEMENTS (dims_nchw), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_if (layout.planar);
  fail_unless_equals_int (layout.channels_dim, 1);
  fail_unless_equals_int (layout.height_dim, 2);
  fail_unless_equals_int (layout.width_dim, 3);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_ccps_rejects_non_tensor_like_formats)
{
  /* Verify formats that can't be mapped onto a tensor are rejected:
   * subsampled, semi-planar and padded */
  const gchar *formats[] = { "I420", "NV12", "BGRx" };
  gsize dims[] = { 4, 8, 3 };
  guint i;

  for (i = 0; i < G_N_ELEMENTS (formats); i++) {
    GstCaps *caps = gst_caps_new_simple ("video/x-raw", "format",
        G_TYPE_STRING, formats[i], NULL);
    GstAnalyticsVideoTensorLayout layout;
    GError *error = NULL;

    fail_if (gst_analytics_video_tensor_layout_from_caps (dims,
            G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
            &error), "%s", formats[i]);
    fail_unless (error != NULL);
    fail_unless_equals_int (error->code,
        GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT);
    g_clear_error (&error);

    gst_caps_unref (caps);
  }
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_rejects_channel_mismatch)
{
  /* Verify dims disagreeing with the declared format's channel count
   * are rejected */
  /* format=RGB expects 3 channels in the channels dim, but dims[2] is 1. */
  gsize dims[] = { 4, 8, 1 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)RGB");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_if (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless (error != NULL);
  fail_unless (error->domain == GST_ANALYTICS_VIDEO_ERROR);
  fail_unless_equals_int (error->code, GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH);
  g_clear_error (&error);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_rejects_unknown_format)
{
  /* Verify an unrecognized declared format string is rejected */
  gsize dims[] = { 4, 8, 3 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)MONKEY");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_if (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT);
  g_clear_error (&error);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_rejects_out_of_range_dims_count)
{
  /* Verify an out-of-range dims count is rejected even with a format
   * declared in caps */
  gsize dims[] = { 4 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)GRAY8");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_if (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS);
  g_clear_error (&error);

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_accepts_matching_declared_resolution)
{
  /* Verify a declared width/height agreeing with dims is accepted */
  gsize dims[] = { 4, 8, 3 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)RGB, "
      "width=(int)8, height=(int)4");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_unless (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_layout_from_caps_rejects_declared_resolution_mismatch)
{
  /* Verify a declared width/height disagreeing with dims is rejected */
  gsize dims[] = { 4, 8, 3 };
  GstCaps *caps = gst_caps_from_string ("video/x-raw, format=(string)RGB, "
      "width=(int)16, height=(int)4");
  GstAnalyticsVideoTensorLayout layout;
  GError *error = NULL;

  fail_if (gst_analytics_video_tensor_layout_from_caps (dims,
          G_N_ELEMENTS (dims), gst_caps_get_structure (caps, 0), &layout,
          &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code, GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH);
  g_clear_error (&error);

  gst_caps_unref (caps);
}

GST_END_TEST;

/* gst_analytics_video_format_guess() */

GST_START_TEST (test_format_guess_uint8_is_identity)
{
  /* Verify uint8 tensors resolve to the same base video format */
  GstVideoFormat out_format;
  GError *error = NULL;

  fail_unless (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_RGB,
          GST_TENSOR_DATA_TYPE_UINT8, &out_format, &error));
  fail_unless_equals_int (out_format, GST_VIDEO_FORMAT_RGB);

  fail_unless (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_GRAY8,
          GST_TENSOR_DATA_TYPE_UINT8, &out_format, &error));
  fail_unless_equals_int (out_format, GST_VIDEO_FORMAT_GRAY8);

  fail_unless (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_RGBP,
          GST_TENSOR_DATA_TYPE_UINT8, &out_format, &error));
  fail_unless_equals_int (out_format, GST_VIDEO_FORMAT_RGBP);
}

GST_END_TEST;

GST_START_TEST (test_format_guess_float32_maps_to_f32_variant)
{
  /* Verify float32 tensors resolve to the format's F32 variant */
  GstVideoFormat out_format;
  GError *error = NULL;

  fail_unless (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_RGB,
          GST_TENSOR_DATA_TYPE_FLOAT32, &out_format, &error));
  fail_unless_equals_int (out_format, GST_ANALYTICS_VIDEO_FORMAT_RGB_F32);

  fail_unless (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_GRAY8,
          GST_TENSOR_DATA_TYPE_FLOAT32, &out_format, &error));
  fail_unless_equals_int (out_format, GST_ANALYTICS_VIDEO_FORMAT_GRAY_F32);

  fail_unless (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_RGBP,
          GST_TENSOR_DATA_TYPE_FLOAT32, &out_format, &error));
  fail_unless_equals_int (out_format, GST_ANALYTICS_VIDEO_FORMAT_RGBP_F32);
}

GST_END_TEST;

GST_START_TEST (test_format_guess_f32_base_format_is_identity)
{
  /* Verify a base format that's already an F32 variant stays unchanged */
  GstVideoFormat out_format;
  GError *error = NULL;

  fail_unless (gst_analytics_video_format_guess
      (GST_ANALYTICS_VIDEO_FORMAT_RGB_F32, GST_TENSOR_DATA_TYPE_FLOAT32,
          &out_format, &error));
  fail_unless_equals_int (out_format, GST_ANALYTICS_VIDEO_FORMAT_RGB_F32);
}

GST_END_TEST;

GST_START_TEST (test_format_guess_rejects_unsupported_combination)
{
  GstVideoFormat out_format;
  GError *error = NULL;

  /* Verify an unsupported data type for a supported base format is
   * rejected */
  /* Remove this test if we add support for u32 bits components */
  fail_if (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_RGB,
          GST_TENSOR_DATA_TYPE_INT32, &out_format, &error));
  fail_unless (error != NULL);
  fail_unless (error->domain == GST_ANALYTICS_VIDEO_ERROR);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT);
  g_clear_error (&error);
}

GST_END_TEST;

GST_START_TEST (test_format_guess_rejects_unhandled_base_format)
{
  /* Verify a base format outside the supported set is rejected */
  GstVideoFormat out_format;
  GError *error = NULL;

  fail_if (gst_analytics_video_format_guess (GST_VIDEO_FORMAT_YUY2,
          GST_TENSOR_DATA_TYPE_UINT8, &out_format, &error));
  fail_unless (error != NULL);
  fail_unless_equals_int (error->code,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT);
  g_clear_error (&error);
}

GST_END_TEST;

/* gst_analytics_video_scales_offsets_are_identity() */

GST_START_TEST (test_scales_offsets_identity_true_for_exact_identity)
{
  /* Verify exact (1.0, 0.0) scale/offset pairs are reported as identity */
  gdouble scales[] = { 1.0, 1.0, 1.0 };
  gdouble offsets[] = { 0.0, 0.0, 0.0 };

  fail_unless (gst_analytics_video_scales_offsets_are_identity (scales,
          offsets, 3, GST_TENSOR_DATA_TYPE_UINT8));
}

GST_END_TEST;

GST_START_TEST (test_scales_offsets_identity_false_when_normalizing)
{
  /* Verify a real normalizing scale is reported as not identity */
  gdouble scales[] = { 1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0 };
  gdouble offsets[] = { 0.0, 0.0, 0.0 };

  fail_if (gst_analytics_video_scales_offsets_are_identity (scales, offsets,
          3, GST_TENSOR_DATA_TYPE_FLOAT32));
}

GST_END_TEST;

GST_START_TEST (test_scales_offsets_identity_false_on_nonzero_offset)
{
  /* Verify a nonzero offset alone is enough to report not identity */
  gdouble scales[] = { 1.0 };
  gdouble offsets[] = { 0.5 };

  fail_if (gst_analytics_video_scales_offsets_are_identity (scales, offsets,
          1, GST_TENSOR_DATA_TYPE_UINT8));
}

GST_END_TEST;

GST_START_TEST (test_scales_offsets_identity_respects_per_type_epsilon)
{
  /* Verify the identity epsilon is tighter for float32 than uint8 */
  /* Within uint8's margin (0.001) of identity. */
  gdouble scales[] = { 1.0001 };
  gdouble offsets[] = { 0.0001 };

  fail_unless (gst_analytics_video_scales_offsets_are_identity (scales,
          offsets, 1, GST_TENSOR_DATA_TYPE_UINT8));

  /* The same values are outside float32's tighter margin (1e-6). */
  fail_if (gst_analytics_video_scales_offsets_are_identity (scales, offsets,
          1, GST_TENSOR_DATA_TYPE_FLOAT32));
}

GST_END_TEST;

static Suite *
analyticsvideoutils_suite (void)
{
  Suite *s = suite_create ("GstAnalyticsVideoUtils");
  TCase *tc_layout_from_dims = tcase_create ("layout-from-dims");
  TCase *tc_layout_from_caps = tcase_create ("layout-from-caps");
  TCase *tc_format_guess = tcase_create ("format-guess");
  TCase *tc_scales_offsets = tcase_create ("scales-offsets-identity");

  suite_add_tcase (s, tc_layout_from_dims);
  tcase_add_test (tc_layout_from_dims, test_layout_from_dims_2d_is_gray8);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_3d_chw_is_planar_rgb);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_3d_hwc_is_interleaved_rgb);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_3d_single_channel_is_gray8);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_4d_nchw_is_planar_with_batch);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_4d_nhwc_is_interleaved_with_batch);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_rejects_out_of_range_dims_count);
  tcase_add_test (tc_layout_from_dims,
      test_layout_from_dims_rejects_unrecognized_channel_count);

  suite_add_tcase (s, tc_layout_from_caps);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_no_format_field_is_unknown);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_resolves_declared_format);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_resolves_declared_planar_format);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_gray_channel_first);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_ccps_rejects_non_tensor_like_formats);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_rejects_channel_mismatch);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_rejects_unknown_format);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_rejects_out_of_range_dims_count);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_accepts_matching_declared_resolution);
  tcase_add_test (tc_layout_from_caps,
      test_layout_from_caps_rejects_declared_resolution_mismatch);

  suite_add_tcase (s, tc_format_guess);
  tcase_add_test (tc_format_guess, test_format_guess_uint8_is_identity);
  tcase_add_test (tc_format_guess,
      test_format_guess_float32_maps_to_f32_variant);
  tcase_add_test (tc_format_guess,
      test_format_guess_f32_base_format_is_identity);

  tcase_add_test (tc_format_guess,
      test_format_guess_rejects_unsupported_combination);
  tcase_add_test (tc_format_guess,
      test_format_guess_rejects_unhandled_base_format);

  suite_add_tcase (s, tc_scales_offsets);
  tcase_add_test (tc_scales_offsets,
      test_scales_offsets_identity_true_for_exact_identity);
  tcase_add_test (tc_scales_offsets,
      test_scales_offsets_identity_false_when_normalizing);
  tcase_add_test (tc_scales_offsets,
      test_scales_offsets_identity_false_on_nonzero_offset);
  tcase_add_test (tc_scales_offsets,
      test_scales_offsets_identity_respects_per_type_epsilon);

  return s;
}

GST_CHECK_MAIN (analyticsvideoutils);
