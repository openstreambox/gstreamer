/* GStreamer
 * Copyright (C) 2026 Collabora Ltd
 * @author: Daniel Morin <daniel.morin@collabora.com>
 *
 * gstanalyticsvideoutils.h
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

#ifndef __GST_ANALYTICS_VIDEO_UTILS_H__
#define __GST_ANALYTICS_VIDEO_UTILS_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/analytics/analytics-meta-prelude.h>
#include <gst/analytics/gsttensor.h>

G_BEGIN_DECLS

/**
 * GST_ANALYTICS_VIDEO_FORMAT_RGB_F32:
 *
 * Native endian variant of %GST_VIDEO_FORMAT_RGB_F32LE or
 * %GST_VIDEO_FORMAT_RGB_F32BE.
 *
 * Since: 1.30
 */
/**
 * GST_ANALYTICS_VIDEO_FORMAT_RGBP_F32:
 *
 * Native endian variant of %GST_VIDEO_FORMAT_RGBP_F32LE or
 * %GST_VIDEO_FORMAT_RGBP_F32BE.
 *
 * Since: 1.30
 */
/**
 * GST_ANALYTICS_VIDEO_FORMAT_GRAY_F32:
 *
 * Native endian variant of %GST_VIDEO_FORMAT_GRAY_F32LE or
 * %GST_VIDEO_FORMAT_GRAY_F32BE.
 *
 * Since: 1.30
 */
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
#define GST_ANALYTICS_VIDEO_FORMAT_RGB_F32   GST_VIDEO_FORMAT_RGB_F32LE
#define GST_ANALYTICS_VIDEO_FORMAT_RGBP_F32  GST_VIDEO_FORMAT_RGBP_F32LE
#define GST_ANALYTICS_VIDEO_FORMAT_GRAY_F32  GST_VIDEO_FORMAT_GRAY_F32LE
#else
#define GST_ANALYTICS_VIDEO_FORMAT_RGB_F32   GST_VIDEO_FORMAT_RGB_F32BE
#define GST_ANALYTICS_VIDEO_FORMAT_RGBP_F32  GST_VIDEO_FORMAT_RGBP_F32BE
#define GST_ANALYTICS_VIDEO_FORMAT_GRAY_F32  GST_VIDEO_FORMAT_GRAY_F32BE
#endif

/**
 * GST_ANALYTICS_VIDEO_ERROR:
 *
 * Error domain for gst_analytics_video_ functions.
 *
 * Since: 1.30
 */
#define GST_ANALYTICS_VIDEO_ERROR (gst_analytics_video_error_quark ())

GST_ANALYTICS_META_API
GQuark gst_analytics_video_error_quark (void);

/**
 * GstAnalyticsVideoError:
 * @GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS: the tensor dims can't be
 *    interpreted as a video frame
 * @GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT: the video format can't be
 *    used for the tensor
 * @GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH: the caps disagree with the tensor
 *    dims
 *
 * Error codes for the %GST_ANALYTICS_VIDEO_ERROR domain.
 *
 * Since: 1.30
 */
typedef enum
{
  GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_DIMS,
  GST_ANALYTICS_VIDEO_ERROR_UNSUPPORTED_FORMAT,
  GST_ANALYTICS_VIDEO_ERROR_CAPS_MISMATCH
} GstAnalyticsVideoError;

/**
 * GstAnalyticsVideoTensorLayout:
 * @format: base #GstVideoFormat declared or guessed for the tensor
 * @width: tensor width, or 0 if not applicable
 * @height: tensor height, or 0 if not applicable
 * @channels: number of channels
 * @planar: %TRUE when video format is planar
 * @batch_dim: index of the batch dimension in the tensor's dims array, or -1
 *    when batch is implicit.
 * @channels_dim: index of the channels dimension, or -1 when the tensor has
 *    none
 * @height_dim: index of the height dimension
 * @width_dim: index of the width dimension
 *
 * Describes how a tensor's dimensions map onto a video frame's layout.
 *
 * Since: 1.30
 */
typedef struct _GstAnalyticsVideoTensorLayout
{
  GstVideoFormat format;
  gint width;
  gint height;
  gint channels;
  gboolean planar;
  gint batch_dim;
  gint channels_dim;
  gint height_dim;
  gint width_dim;

  /*< private >*/
  gpointer _gst_reserved[GST_PADDING];
} GstAnalyticsVideoTensorLayout;

GST_ANALYTICS_META_API
gboolean gst_analytics_video_tensor_layout_from_dims (const gsize * dims,
    gsize dims_count, GstAnalyticsVideoTensorLayout * layout,
    GError ** error);

GST_ANALYTICS_META_API
gboolean gst_analytics_video_tensor_layout_from_caps (const gsize * dims,
    gsize dims_count, const GstStructure * s,
    GstAnalyticsVideoTensorLayout * layout, GError ** error);

GST_ANALYTICS_META_API
gboolean gst_analytics_video_format_guess (GstVideoFormat base_format,
    GstTensorDataType data_type, GstVideoFormat * out_format,
    GError ** error);


GST_ANALYTICS_META_API
gboolean gst_analytics_video_scales_offsets_are_identity (const gdouble * scales,
    const gdouble * offsets, guint n_channels, GstTensorDataType data_type);

G_END_DECLS

#endif /* __GST_ANALYTICS_VIDEO_UTILS_H__ */
