/* GStreamer
 * Copyright (C) 2026 Collabora Ltd
 * @author: Daniel Morin <daniel.morin@collabora.com>
 *
 * gstanalyticsvideoutils.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstanalyticsvideoutils.h"
#include "modelinfo.h"
#include <math.h>
#include <string.h>

/* Largest deviation from (scale 1.0, offset 0.0) still treated as identity. */
#define NORMALIZATION_EPSILON_UINT8  0.001
#define NORMALIZATION_EPSILON_FLOAT  1e-6

/**
 * gst_analytics_video_error_quark:
 *
 * Error domain for gst_analytics_video_ functions.
 *
 * Since: 1.30
 */
GQuark
gst_analytics_video_error_quark (void)
{
  return g_quark_from_static_string ("gst-analytics-video-error-quark");
}

static void
gst_analytics_video_tensor_layout_init (GstAnalyticsVideoTensorLayout * layout)
{
  layout->format = GST_VIDEO_FORMAT_UNKNOWN;
  layout->width = 0;
  layout->height = 0;
  layout->channels = 0;
  layout->planar = FALSE;
  layout->batch_dim = -1;
  layout->channels_dim = -1;
  layout->height_dim = -1;
  layout->width_dim = -1;

  memset (layout->_gst_reserved, 0, sizeof (layout->_gst_reserved));
}

/**
 * gst_analytics_video_tensor_layout_from_dims:
 * @dims: (array length=dims_count): the tensor's dimensions
 * @dims_count: number of entries in @dims
 * @layout: (out caller-allocates): filled in with the guessed layout
 * @error: return location for a #GError, or %NULL
 *
 * Guesses a #GstAnalyticsVideoTensorLayout purely from a tensor's shape
 * (2 to 4 dimensions), without any prior knowledge of its intended video
 * format. On success use gst_analytics_video_format_guess() afterward to map
 * to a float32 variant when the tensor's data type calls for one.
 *
 * Returns: %TRUE on success, %FALSE with @error set if @dims_count is out
 *    of range or the dims can't be interpreted as a video layout
 *
 * Since: 1.30
 */
gboolean
gst_analytics_video_tensor_layout_from_dims (const gsize * dims,
    gsize dims_count, GstAnalyticsVideoTensorLayout * layout, GError ** error)
{
  gst_analytics_video_tensor_layout_init (layout);

  if (dims_count < 2 || dims_count > 4) {
    g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
        GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS,
        "Don't know how to interpret tensors with %" G_GSIZE_FORMAT
        " dimensions", dims_count);
    return FALSE;
  }

  switch (dims_count) {
    case 2:
      layout->format = GST_VIDEO_FORMAT_GRAY8;
      layout->height_dim = 0;
      layout->width_dim = 1;
      layout->height = dims[0];
      layout->width = dims[1];
      layout->channels = 1;
      break;
    case 3:
      if (dims[0] == 1 || dims[0] == 3) {
        layout->channels_dim = 0;
        layout->channels = dims[0];
        if (dims[0] == 1) {
          layout->format = GST_VIDEO_FORMAT_GRAY8;
        } else {
          layout->format = GST_VIDEO_FORMAT_RGBP;
          layout->planar = TRUE;
        }
        layout->height_dim = 1;
        layout->width_dim = 2;
        layout->height = dims[1];
        layout->width = dims[2];
      } else if (dims[2] == 1 || dims[2] == 3) {
        layout->channels_dim = 2;
        layout->channels = dims[2];
        if (dims[2] == 1) {
          layout->format = GST_VIDEO_FORMAT_GRAY8;
        } else {
          layout->format = GST_VIDEO_FORMAT_RGB;
        }
        layout->height_dim = 0;
        layout->width_dim = 1;
        layout->height = dims[0];
        layout->width = dims[1];
      } else {
        g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
            GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS,
            "Don't know how to interpret dims");
        return FALSE;
      }
      break;
    case 4:
      /* Assuming dims[0] is a batch */
      layout->batch_dim = 0;
      if (dims[1] == 1 || dims[1] == 3) {
        layout->channels_dim = 1;
        layout->height_dim = 2;
        layout->width_dim = 3;
        layout->planar = TRUE;
      } else if (dims[3] == 1 || dims[3] == 3) {
        layout->height_dim = 1;
        layout->width_dim = 2;
        layout->channels_dim = 3;
      } else {
        g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
            GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS,
            "Don't know how to interpret dims");
        return FALSE;
      }

      layout->channels = dims[layout->channels_dim];
      layout->height = dims[layout->height_dim];
      layout->width = dims[layout->width_dim];

      if (layout->channels == 1) {
        layout->format = GST_VIDEO_FORMAT_GRAY8;
        layout->planar = FALSE;
      } else if (layout->channels == 3) {
        layout->format =
            layout->planar ? GST_VIDEO_FORMAT_RGBP : GST_VIDEO_FORMAT_RGB;
      } else {
        g_assert_not_reached ();
      }
      break;
  }

  return TRUE;
}

/* Whether pixels of @info are laid out like a tensor: no subsampling, and
 * either all components packed in a single plane without padding, or one
 * plane per component. */
static gboolean
format_maps_onto_tensor (const GstVideoFormatInfo * info, gboolean * planar)
{
  guint n_comps = GST_VIDEO_FORMAT_INFO_N_COMPONENTS (info);
  guint n_planes = GST_VIDEO_FORMAT_INFO_N_PLANES (info);
  guint c;

  if (GST_VIDEO_FORMAT_INFO_IS_COMPLEX (info) ||
      GST_VIDEO_FORMAT_INFO_IS_TILED (info) ||
      GST_VIDEO_FORMAT_INFO_HAS_PALETTE (info))
    return FALSE;

  for (c = 0; c < n_comps; c++) {
    if (GST_VIDEO_FORMAT_INFO_W_SUB (info, c) != 0 ||
        GST_VIDEO_FORMAT_INFO_H_SUB (info, c) != 0 ||
        GST_VIDEO_FORMAT_INFO_DEPTH (info, c) % 8 != 0)
      return FALSE;
  }

  if (n_planes == 1) {
    *planar = FALSE;
    return GST_VIDEO_FORMAT_INFO_PSTRIDE (info, 0) ==
        n_comps * GST_VIDEO_FORMAT_INFO_DEPTH (info, 0) / 8;
  }

  *planar = TRUE;
  return n_planes == n_comps;
}

/**
 * gst_analytics_video_tensor_layout_from_caps:
 * @dims: (array length=dims_count): the tensor's dimensions
 * @dims_count: number of entries in @dims
 * @s: the `video/x-raw` #GstStructure declaring the tensor's format.
 * @layout: (out caller-allocates): filled in with the guessed layout,
 *    `layout->format` is left as %GST_VIDEO_FORMAT_UNKNOWN when @s
 *    declares no `format` field.
 * @error: return location for a #GError, or %NULL
 *
 * Guesses a #GstAnalyticsVideoTensorLayout from a video format already
 * declared in @s, validating that @dims agrees with the declared format's
 * channel count and resolution.
 *
 * Only formats whose memory layout maps directly onto a tensor are supported:
 * no chroma subsampling, and either a single plane holding all components with
 * no padding between pixels, or one plane per component.
 *
 * Returns: %TRUE on success, or when @s declares no format at all, otherwise
 *    %FALSE with @error set if @s declares an unknown or unsupported format, or
 *    one that disagrees with @dims
 *
 * Since: 1.30
 */
gboolean
gst_analytics_video_tensor_layout_from_caps (const gsize * dims,
    gsize dims_count, const GstStructure * s,
    GstAnalyticsVideoTensorLayout * layout, GError ** error)
{
  const gchar *format = gst_structure_get_string (s, "format");
  GstVideoFormat vfmt;
  const GstVideoFormatInfo *info;
  guint channels;

  gst_analytics_video_tensor_layout_init (layout);

  if (!format)
    return TRUE;

  vfmt = gst_video_format_from_string (format);
  if (vfmt == GST_VIDEO_FORMAT_UNKNOWN) {
    g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
        GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT,
        "Unknown video format '%s'", format);
    return FALSE;
  }

  info = gst_video_format_get_info (vfmt);
  channels = GST_VIDEO_FORMAT_INFO_N_COMPONENTS (info);

  if (!format_maps_onto_tensor (info, &layout->planar)) {
    g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
        GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT,
        "Video format '%s' can't be mapped onto a tensor", format);
    return FALSE;
  }

  layout->format = vfmt;
  layout->channels = channels;

  if (dims_count < 2 || dims_count > 4) {
    g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
        GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS,
        "Don't know how to interpret tensors with %" G_GSIZE_FORMAT
        " dimensions", dims_count);
    return FALSE;
  }

  switch (dims_count) {
    case 2:
      if (channels != 1) {
        g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
            GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH,
            "Tensors with 2 dimensions have no room for the %u channels of "
            "format '%s'", channels, format);
        return FALSE;
      }
      layout->height_dim = 0;
      layout->width_dim = 1;
      break;
    case 3:
      /* A single channel can be either first or last */
      if (layout->planar || (channels == 1 && dims[0] == 1)) {
        layout->channels_dim = 0;
        layout->height_dim = 1;
        layout->width_dim = 2;
      } else {
        layout->channels_dim = 2;
        layout->height_dim = 0;
        layout->width_dim = 1;
      }
      break;
    case 4:
      /* Assuming dims[0] is a batch */
      layout->batch_dim = 0;
      if (layout->planar || (channels == 1 && dims[1] == 1)) {
        layout->channels_dim = 1;
        layout->height_dim = 2;
        layout->width_dim = 3;
      } else {
        layout->height_dim = 1;
        layout->width_dim = 2;
        layout->channels_dim = 3;
      }
      break;
  }

  if (layout->channels_dim >= 0 && dims[layout->channels_dim] != channels) {
    g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
        GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH,
        "Dims don't have %u channels in the dimension expected for "
        "format '%s'", channels, format);
    return FALSE;
  }

  layout->height = dims[layout->height_dim];
  layout->width = dims[layout->width_dim];

  if (!gst_analytics_modelinfo_validate_video_caps_resolution (s,
          layout->width, layout->height)) {
    g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
        GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH,
        "Caps declared width/height inconsistent with the " "tensor's dims");
    return FALSE;
  }

  return TRUE;
}

/**
 * gst_analytics_video_format_guess:
 * @base_format: the format from gst_analytics_video_tensor_layout_from_dims()
 *    or gst_analytics_video_tensor_layout_from_caps()
 * @data_type: the tensor's actual element data type
 * @out_format: (out): the concrete #GstVideoFormat to negotiate
 * @error: return location for a #GError, or %NULL
 *
 * Maps @base_format to the concrete #GstVideoFormat matching @data_type:
 * unchanged for %GST_TENSOR_DATA_TYPE_UINT8, or the corresponding
 * `*_F32LE`/`*_F32BE` variant for %GST_TENSOR_DATA_TYPE_FLOAT32.
 *
 * Returns: %TRUE on success, %FALSE with @error set if @data_type has no
 *    supported video format for @base_format
 *
 * Since: 1.30
 */
gboolean
gst_analytics_video_format_guess (GstVideoFormat base_format,
    GstTensorDataType data_type, GstVideoFormat * out_format, GError ** error)
{
  switch (base_format) {
    case GST_VIDEO_FORMAT_GRAY8:
      if (data_type == GST_TENSOR_DATA_TYPE_UINT8) {
        *out_format = GST_VIDEO_FORMAT_GRAY8;
      } else if (data_type == GST_TENSOR_DATA_TYPE_FLOAT32) {
        *out_format = GST_ANALYTICS_VIDEO_FORMAT_GRAY_F32;
      } else {
        goto unsupported;
      }
      break;
    case GST_VIDEO_FORMAT_RGB:
      if (data_type == GST_TENSOR_DATA_TYPE_UINT8) {
        *out_format = GST_VIDEO_FORMAT_RGB;
      } else if (data_type == GST_TENSOR_DATA_TYPE_FLOAT32) {
        *out_format = GST_ANALYTICS_VIDEO_FORMAT_RGB_F32;
      } else {
        goto unsupported;
      }
      break;
    case GST_VIDEO_FORMAT_RGBP:
      if (data_type == GST_TENSOR_DATA_TYPE_UINT8) {
        *out_format = GST_VIDEO_FORMAT_RGBP;
      } else if (data_type == GST_TENSOR_DATA_TYPE_FLOAT32) {
        *out_format = GST_ANALYTICS_VIDEO_FORMAT_RGBP_F32;
      } else {
        goto unsupported;
      }
      break;
    default:
      if ((base_format == GST_ANALYTICS_VIDEO_FORMAT_GRAY_F32
              || base_format == GST_ANALYTICS_VIDEO_FORMAT_RGB_F32
              || base_format == GST_ANALYTICS_VIDEO_FORMAT_RGBP_F32)
          && data_type == GST_TENSOR_DATA_TYPE_FLOAT32) {
        *out_format = base_format;
      } else {
        goto unsupported;
      }
      break;
  }

  return TRUE;

unsupported:
  g_set_error (error, GST_ANALYTICS_VIDEO_ERROR,
      GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT,
      "Unsupported format '%s' for tensor data type",
      gst_video_format_to_string (base_format));
  return FALSE;
}

/**
 * gst_analytics_video_scales_offsets_are_identity:
 * @scales: (array length=n_channels): per-channel scale factors
 * @offsets: (array length=n_channels): per-channel offsets
 * @n_channels: number of channels
 * @data_type: the tensor's data type, selecting the epsilon within which a
 *    `(scale, offset)` pair is treated as `(1.0, 0.0)`
 *
 * Checks whether every channel's `(scale, offset)` pair is close enough
 * to `(1.0, 0.0)` to be treated as a no-op normalization.
 *
 * A channel is an identity when both `|scale - 1.0|` and `|offset|` are at
 * most an epsilon that depends on @data_type:
 *
 * - %GST_TENSOR_DATA_TYPE_UINT8: Epsilon is 0.001.
 * - Any other data type: Epsilon is 1e-6.
 *
 * Returns: %TRUE if normalization would be a no-op for all channels
 *
 * Since: 1.30
 */
gboolean
gst_analytics_video_scales_offsets_are_identity (const gdouble * scales,
    const gdouble * offsets, guint n_channels, GstTensorDataType data_type)
{
  gdouble epsilon = data_type == GST_TENSOR_DATA_TYPE_UINT8 ?
      NORMALIZATION_EPSILON_UINT8 : NORMALIZATION_EPSILON_FLOAT;
  guint i;

  for (i = 0; i < n_channels; i++) {
    if (fabs (scales[i] - 1.0) > epsilon || fabs (offsets[i]) > epsilon)
      return FALSE;
  }

  return TRUE;
}
