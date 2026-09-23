/* GStreamer
 * Copyright (C) 2026 Collabora Ltd
 * @author: Daniel Morin <daniel.morin@collabora.com>
 *
 * gstanalyticsinferenceutils.h
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

#ifndef __GST_ANALYTICS_INFERENCE_UTILS_H__
#define __GST_ANALYTICS_INFERENCE_UTILS_H__

#include <gst/gst.h>
#include <gst/analytics/analytics-meta-prelude.h>

G_BEGIN_DECLS

GST_ANALYTICS_META_API
GstCaps *gst_analytics_inference_transform_sink_caps (GstCaps * caps,
    GstCaps * restrictions, const GstStructure * model_tensorgroups,
    const gchar * group_id);

GST_ANALYTICS_META_API
GstCaps *gst_analytics_inference_transform_src_caps (GstCaps * caps,
    GstCaps * restrictions);

GST_ANALYTICS_META_API
gboolean gst_analytics_inference_caps_has_tensor_group (const GstCaps * caps,
    const gchar * group_id);

G_END_DECLS

#endif /* __GST_ANALYTICS_INFERENCE_UTILS_H__ */
