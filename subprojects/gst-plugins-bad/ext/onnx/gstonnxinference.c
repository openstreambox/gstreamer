/*
 * GStreamer gstreamer-onnxinference
 * Copyright (C) 2023-2025 Collabora Ltd.
 * Copyright (C) 2026 Seungha Yang <seungha@centricular.com>
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

/**
 * SECTION:element-onnxinference
 * @short_description: Run ONNX inference model on video buffers
 *
 * This element can apply an ONNX model to video buffers. It attaches
 * the tensor output to the buffer as a @ref GstTensorMeta.
 *
 * To install ONNX on your system, follow the instructions in the
 * README.md in with this plugin.
 *
 * ## Example launch line
 *
 * Test image file, model file (SSD) and label file can be found here:
 * https://gitlab.collabora.com/gstreamer/onnx-models
 *
 * |[
 * GST_DEBUG=ssdobjectdetector:5 \
 * gst-launch-1.0 filesrc location=onnx-models/images/bus.jpg ! \
 * jpegdec ! videoconvert ! onnxinference execution-provider=cpu model-file=onnx-models/models/ssd_mobilenet_v1_coco.onnx ! \
 * ssdobjectdetector label-file=onnx-models/labels/COCO_classes.txt ! videoconvert ! imagefreeze ! autovideosink
 * ]|
 *
 * Note: in order for downstream tensor decoders to correctly parse the tensor
 * data in the GstTensorMeta, meta data must be attached to tensors. The
 * inference element gets this model metadata from the modelinfo file annexed
 * to the model. The modelinfo-helper tool can be used to create a modelinfo
 * file: https://gitlab.freedesktop.org/gstreamer/gstreamer/-/tree/main/subprojects/gst-devtools/modelinfo-helper
 *
 * ## Modelinfo example for ssd_mobilenet_v1_coco.onnx
 *
 * |[
 * [modelinfo]
 * version=1.0
 * group-id=ssd-mobilenet-v1-variant-1-out
 *
 * [image_tensor:0]
 * id=image_tensor_0
 * type=uint8
 * dims=-1,-1,-1,3
 * dir=input
 * ranges=0.0,255.0;0.0,255.0;0.0,255.0
 *
 * [detection_boxes:0]
 * id=ssd-mobilenet-v1-variant-1-out-boxes
 * type=float32
 * dims=-1,-1,4
 * dir=output
 *
 * [detection_classes:0]
 * id=ssd-mobilenet-v1-variant-1-out-classes
 * type=float32
 * dims=-1,-1
 * dir=output
 *
 * [detection_scores:0]
 * id=ssd-mobilenet-v1-variant-1-out-scores
 * type=float32
 * dims=-1,-1
 * dir=output
 *
 * [num_detections:0]
 * id=generic-variant-1-out-count
 * type=float32
 * dims=-1
 * dir=output
 * ]|
 *
 * The modelinfo file should be placed alongside the model file with a
 * `.modelinfo` suffix appended to the model filename. For example:
 *
 * |[
 * /path/to/model.onnx
 * /path/to/model.onnx.modelinfo
 * ]|
 *
 * As a convenience, sample models with their modelinfo files are available
 * here: https://gitlab.collabora.com/gstreamer/onnx-models/-/tree/master/models
 *
 * Since: 1.20
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gstonnx-config.h"
#include "gstonnxinference.h"
#if HAVE_DIRECTML
#include "gstonnximporter-dml.h"
#else
#define GST_CAPS_FEATURE_MEMORY_D3D12_MEMORY "memory:D3D12Memory"
#endif
#include "gstonnximporter.h"

#include <gst/gst.h>
#include <gst/video/video.h>
#include <string.h>



#if HAVE_GST_HIP
#include <gst/hip/gsthip.h>
#include "gstonnximporter-hip.h"
#else
/* define to reduce ifdefs */
#define GST_CAPS_FEATURE_MEMORY_HIP_MEMORY "memory:HIPMemory"
#endif

GST_DEBUG_CATEGORY (onnx_inference_debug);
#define GST_CAT_DEFAULT onnx_inference_debug

/* TODO: Add F16 formats */
#if G_BYTE_ORDER == G_LITTLE_ENDIAN
#define GST_ONNX_VIDEO_FORMATS \
    "{ RGB, RGBP, GRAY8, RGB_F32LE, RGBP_F32LE, GRAY_F32LE }"
#else
#define GST_ONNX_VIDEO_FORMATS \
    "{ RGB, RGBP, GRAY8, RGB_F32BE, RGBP_F32BE, GRAY_F32BE }"
#endif

#define DOC_CAPS_STR \
  GST_VIDEO_CAPS_MAKE (GST_ONNX_VIDEO_FORMATS) "; " \
  GST_VIDEO_CAPS_MAKE_WITH_FEATURES (GST_CAPS_FEATURE_MEMORY_HIP_MEMORY, \
      GST_ONNX_VIDEO_FORMATS) "; " \
  GST_VIDEO_CAPS_MAKE_WITH_FEATURES (GST_CAPS_FEATURE_MEMORY_D3D12_MEMORY, \
      GST_ONNX_VIDEO_FORMATS)

#if HAVE_GST_HIP
static gboolean have_hip_rtc = FALSE;
#endif

struct _GstOnnxInference
{
  GstOnnxBaseInference parent;

  GstVideoInfo video_info;

  /* Raw dimensions returned from OrtApi::GetDimensions() */
  gint64 input_dims_model[GST_ONNX_MAX_DIMS];

  gint model_in_width;
  gint model_in_height;
  gint channels;
  gboolean planar;
  gint height_dim;
  gint width_dim;
  gint channels_dim;
  gint batch_dim;

  GstCaps *input_tensors_caps;
  GstCaps *output_tensors_caps;
  gchar *group_id;

  GstOnnxImporterConfig importer_config;
  GstOnnxImporter *importer;

  GMutex lock;
};

static void gst_onnx_inference_finalize (GObject * object);

static GstCaps *gst_onnx_inference_transform_caps (GstBaseTransform *
    trans, GstPadDirection direction, GstCaps * caps, GstCaps * filter_caps);
static gboolean
gst_onnx_inference_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps);
static gboolean
gst_onnx_inference_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query);

static void gst_onnx_inference_reset (GstOnnxBaseInference * base);
static gboolean gst_onnx_inference_process_model_info (GstOnnxBaseInference *
    base, GstAnalyticsModelInfo * modelinfo,
    const GstOnnxModelInfo * parsed_info, GstOnnxImporter * importer);
static GstFlowReturn
gst_onnx_inference_prepare_input_tensors (GstOnnxBaseInference * base,
    GstBuffer * buffer, const GArray * model_inputs, gsize iteration,
    GPtrArray * input_tensors);
static void gst_onnx_inference_unprepare_input_tensors (GstOnnxBaseInference *
    base, GstBuffer * buffer, GPtrArray * input_tensors);

#define gst_onnx_inference_parent_class parent_class
G_DEFINE_TYPE (GstOnnxInference, gst_onnx_inference,
    GST_TYPE_ONNX_BASE_INFERENCE);
GST_ELEMENT_REGISTER_DEFINE (onnx_inference, "onnxinference", GST_RANK_PRIMARY,
    GST_TYPE_ONNX_INFERENCE);

static void
gst_onnx_inference_class_init (GstOnnxInferenceClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstOnnxBaseInferenceClass *onnx_class = GST_ONNX_BASE_INFERENCE_CLASS (klass);
  GstPadTemplate *pad_templ;
  GstCaps *templ_caps;
  GstCaps *doc_caps;

  gobject_class->finalize = gst_onnx_inference_finalize;

  gst_element_class_set_static_metadata (element_class, "onnxinference",
      "Filter/Video",
      "Apply neural network to video frames and create tensor output",
      "Aaron Boxer <aaron.boxer@collabora.com>");

  templ_caps =
      gst_caps_from_string (GST_VIDEO_CAPS_MAKE (GST_ONNX_VIDEO_FORMATS));
#if HAVE_DIRECTML
  {
    GstCaps *d3d12_caps = gst_caps_copy (templ_caps);
    gst_caps_set_features_simple (d3d12_caps,
        gst_caps_features_new_single_static_str
        (GST_CAPS_FEATURE_MEMORY_D3D12_MEMORY));
    gst_caps_append (templ_caps, d3d12_caps);
  }
#endif
#if HAVE_GST_HIP
  if (gst_hip_rtc_load_library (GST_HIP_VENDOR_AMD)) {
    GstCaps *hip_caps = gst_caps_copy (templ_caps);
    gst_caps_set_features_simple (hip_caps,
        gst_caps_features_new_single_static_str
        (GST_CAPS_FEATURE_MEMORY_HIP_MEMORY));
    gst_caps_append (templ_caps, hip_caps);
    have_hip_rtc = TRUE;
  }
#endif

  pad_templ = gst_pad_template_new ("sink",
      GST_PAD_SINK, GST_PAD_ALWAYS, templ_caps);
  doc_caps = gst_caps_from_string (DOC_CAPS_STR);
  gst_pad_template_set_documentation_caps (pad_templ, doc_caps);
  gst_element_class_add_pad_template (element_class, pad_templ);

  pad_templ = gst_pad_template_new ("src",
      GST_PAD_SRC, GST_PAD_ALWAYS, templ_caps);
  gst_pad_template_set_documentation_caps (pad_templ, doc_caps);
  gst_element_class_add_pad_template (element_class, pad_templ);
  gst_caps_unref (doc_caps);
  gst_caps_unref (templ_caps);

  gst_element_class_mark_as_untrusted_aware (element_class);

  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_onnx_inference_transform_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_onnx_inference_set_caps);
  trans_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_onnx_inference_propose_allocation);

  onnx_class->reset = GST_DEBUG_FUNCPTR (gst_onnx_inference_reset);
  onnx_class->process_model_info =
      GST_DEBUG_FUNCPTR (gst_onnx_inference_process_model_info);
  onnx_class->prepare_input_tensors =
      GST_DEBUG_FUNCPTR (gst_onnx_inference_prepare_input_tensors);
  onnx_class->unprepare_input_tensors =
      GST_DEBUG_FUNCPTR (gst_onnx_inference_unprepare_input_tensors);
  onnx_class->proc_type = GST_ONNX_IMPORTER_PROC_CONVERT;

  GST_DEBUG_CATEGORY_INIT (onnx_inference_debug, "onnxinference", 0,
      "ONNX inference");
}

static void
gst_onnx_inference_init (GstOnnxInference * self)
{
  gst_video_info_init (&self->video_info);

  self->input_tensors_caps = gst_caps_new_empty_simple ("video/x-raw");
  self->output_tensors_caps = gst_caps_new_empty_simple ("video/x-raw");

  self->height_dim = -1;
  self->width_dim = -1;
  self->channels_dim = -1;
  self->batch_dim = -1;

  g_mutex_init (&self->lock);
}

static void
gst_onnx_inference_finalize (GObject * object)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (object);

  g_mutex_clear (&self->lock);
  gst_clear_caps (&self->input_tensors_caps);
  gst_clear_caps (&self->output_tensors_caps);
  g_free (self->group_id);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

static gboolean
gst_onnx_inference_can_use_hip (GstOnnxInference * self)
{
#if HAVE_GST_HIP
  GstOnnxBaseInference *base = GST_ONNX_BASE_INFERENCE (self);

  if (!have_hip_rtc ||
      (base->execution_provider != GST_ONNX_EXECUTION_PROVIDER_MIGRAPHX &&
          base->execution_provider != GST_ONNX_EXECUTION_PROVIDER_HIP)) {
    return FALSE;
  }

  if (!self->importer)
    return TRUE;

  if (!GST_IS_ONNX_IMPORTER_HIP (self->importer))
    return FALSE;

  return TRUE;
#else
  return FALSE;
#endif
}

static gboolean
gst_onnx_inference_can_use_dml (GstOnnxInference * self)
{
#if HAVE_DIRECTML
  GstOnnxBaseInference *base = GST_ONNX_BASE_INFERENCE (self);

  return base->execution_provider == GST_ONNX_EXECUTION_PROVIDER_DIRECTML &&
      (!self->importer || GST_IS_ONNX_IMPORTER_DML (self->importer));
#else
  return FALSE;
#endif
}

/* Only remove non-suppported caps features */
static GstCaps *
gst_onnx_inference_filter_memory_caps (GstOnnxInference * self, GstCaps * caps)
{
  GstCaps *ret = gst_caps_new_empty ();
  gboolean have_hip = gst_onnx_inference_can_use_hip (self);
  gboolean have_dml = gst_onnx_inference_can_use_dml (self);
  guint i;

  for (i = 0; i < gst_caps_get_size (caps); i++) {
    GstCapsFeatures *features = gst_caps_get_features (caps, i);
    gboolean is_hip = gst_caps_features_contains (features,
        GST_CAPS_FEATURE_MEMORY_HIP_MEMORY);

    if (is_hip && !have_hip)
      continue;

    if (gst_caps_features_contains (features,
            GST_CAPS_FEATURE_MEMORY_D3D12_MEMORY) && !have_dml)
      continue;

    gst_caps_append_structure_full (ret,
        gst_structure_copy (gst_caps_get_structure (caps, i)),
        gst_caps_features_copy (features));
  }

  return ret;
}

/* Add supported memory caps features */
static GstCaps *
gst_onnx_inference_add_memory_caps (GstOnnxInference * self, GstCaps * caps)
{
  GstCaps *ret = gst_caps_copy (caps);

#if HAVE_DIRECTML
  if (gst_onnx_inference_can_use_dml (self)) {
    GstCaps *d3d12_caps = gst_caps_copy (caps);
    gst_caps_set_features_simple (d3d12_caps,
        gst_caps_features_new_single_static_str
        (GST_CAPS_FEATURE_MEMORY_D3D12_MEMORY));
    gst_caps_append (ret, d3d12_caps);
  }
#endif

#if HAVE_GST_HIP
  if (gst_onnx_inference_can_use_hip (self)) {
    GstCaps *hip_caps = gst_caps_copy (caps);

    gst_caps_set_features_simple (hip_caps,
        gst_caps_features_new_single_static_str
        (GST_CAPS_FEATURE_MEMORY_HIP_MEMORY));
    gst_caps_append (ret, hip_caps);
  }
#endif

  return ret;
}

static GstCaps *
gst_onnx_inference_transform_caps (GstBaseTransform *
    trans, GstPadDirection direction, GstCaps * caps, GstCaps * filter_caps)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (trans);
  GstCaps *other_caps;
  GstCaps *restrictions;

  g_mutex_lock (&self->lock);

  if (!self->importer) {
    other_caps = gst_onnx_inference_filter_memory_caps (self, caps);
    goto done;
  }

  GST_LOG_OBJECT (self, "transforming caps %" GST_PTR_FORMAT, caps);

  GST_DEBUG_OBJECT (self, "Applying model input tensors caps restrictions: %"
      GST_PTR_FORMAT, self->input_tensors_caps);

  restrictions = gst_onnx_inference_add_memory_caps (self,
      self->input_tensors_caps);

  if (direction == GST_PAD_SINK) {
    const GValue *tensors = gst_structure_get_value
        (gst_caps_get_structure (self->output_tensors_caps, 0), "tensors");

    other_caps = gst_analytics_inference_transform_sink_caps (caps,
        restrictions, tensors ? gst_value_get_structure (tensors) : NULL,
        self->group_id);
  } else {
    other_caps = gst_analytics_inference_transform_src_caps (caps,
        restrictions);
  }

  gst_caps_unref (restrictions);

done:
  g_mutex_unlock (&self->lock);

  if (filter_caps) {
    GstCaps *tmp = gst_caps_intersect_full (other_caps, filter_caps,
        GST_CAPS_INTERSECT_FIRST);
    gst_caps_replace (&other_caps, tmp);
    gst_caps_unref (tmp);
  }

  return other_caps;
}

static gsize
get_tensor_type_size (GstTensorDataType data_type)
{
  switch (data_type) {
    case GST_TENSOR_DATA_TYPE_UINT8:
      return sizeof (uint8_t);
    case GST_TENSOR_DATA_TYPE_UINT16:
      return sizeof (uint16_t);
    case GST_TENSOR_DATA_TYPE_UINT32:
      return sizeof (uint32_t);
    case GST_TENSOR_DATA_TYPE_INT32:
      return sizeof (int32_t);
    case GST_TENSOR_DATA_TYPE_FLOAT16:
      return 2;
    case GST_TENSOR_DATA_TYPE_FLOAT32:
      return sizeof (float);
    default:
      return 0;
  };
}

static gboolean
gst_onnx_inference_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (trans);
  GstOnnxImporterConfig *config = &self->importer_config;

  if (!self->importer) {
    GST_ERROR_OBJECT (self, "Importer is not prepared");
    return FALSE;
  }

  if (gst_analytics_inference_caps_has_tensor_group (incaps, self->group_id)) {
    GST_ERROR_OBJECT (self, "Tensor group '%s' is already present upstream",
        self->group_id);
    return FALSE;
  }

  if (!gst_video_info_from_caps (&self->video_info, incaps)) {
    GST_ERROR_OBJECT (self, "Failed to parse caps");
    return FALSE;
  }

  gsize element_size = get_tensor_type_size (config->data_type);
  gsize input_tensor_size;

  g_assert (element_size != 0);

  /* Use GLib's checked multiplication to prevent overflow */
  if (!g_size_checked_mul (&input_tensor_size, self->video_info.width,
          self->video_info.height) ||
      !g_size_checked_mul (&input_tensor_size, input_tensor_size,
          self->channels)
      || !g_size_checked_mul (&input_tensor_size, input_tensor_size,
          element_size)) {
    GST_ERROR_OBJECT (self,
        "Integer overflow in input tensor size: %dx%d pixels, %u channels, %zu bytes per element",
        self->video_info.width, self->video_info.height, self->channels,
        element_size);
    return FALSE;
  }

  /* Resolve dynamic input dimensions and validate fixed ones */
  memcpy (config->dims, self->input_dims_model, sizeof (config->dims));

  if (self->batch_dim >= 0)
    config->dims[self->batch_dim] = 1;

  if (config->dims[self->height_dim] > 0) {
    if (config->dims[self->height_dim] != self->video_info.height) {
      GST_ERROR_OBJECT (self, "Caps have height %d, but model expects %"
          G_GINT64_FORMAT, self->video_info.height,
          config->dims[self->height_dim]);
      return FALSE;
    }
  } else {
    config->dims[self->height_dim] = self->video_info.height;
  }

  if (config->dims[self->width_dim] > 0) {
    if (config->dims[self->width_dim] != self->video_info.width) {
      GST_ERROR_OBJECT (self, "Caps have width %d, but model expects %"
          G_GINT64_FORMAT, self->video_info.width,
          config->dims[self->width_dim]);
      return FALSE;
    }
  } else {
    config->dims[self->width_dim] = self->video_info.width;
  }

  config->tensor_size = input_tensor_size;
  if (!gst_onnx_importer_setup (self->importer, &self->video_info, config)) {
    GST_ERROR_OBJECT (self, "Couldn't setup importer");
    return FALSE;
  }

  return TRUE;
}

static gboolean
gst_onnx_inference_propose_allocation (GstBaseTransform * trans,
    GstQuery * decide_query, GstQuery * query)
{
  if (!GST_BASE_TRANSFORM_CLASS (parent_class)->propose_allocation (trans,
          decide_query, query))
    return FALSE;

  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);

  return TRUE;
}

static void
gst_onnx_inference_reset (GstOnnxBaseInference * base)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (base);

  g_mutex_lock (&self->lock);
  gst_clear_object (&self->importer);
  g_clear_pointer (&self->group_id, g_free);

  memset (&self->importer_config, 0, sizeof (self->importer_config));

  gst_caps_unref (self->input_tensors_caps);
  self->input_tensors_caps = gst_caps_new_empty_simple ("video/x-raw");
  gst_caps_unref (self->output_tensors_caps);
  self->output_tensors_caps = gst_caps_new_empty_simple ("video/x-raw");
  g_mutex_unlock (&self->lock);

  self->model_in_width = 0;
  self->model_in_height = 0;
}

static gboolean
gst_onnx_inference_process_model_info (GstOnnxBaseInference * base,
    GstAnalyticsModelInfo * modelinfo, const GstOnnxModelInfo * parsed_info,
    GstOnnxImporter * importer)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (base);
  const GstOnnxModelInput *input;
  GstOnnxImporterConfig *config = &self->importer_config;
  GstVideoFormat gst_format = GST_VIDEO_FORMAT_UNKNOWN;
  GstAnalyticsVideoTensorLayout layout;
  gboolean layout_ok = TRUE;
  gboolean format_from_modelinfo = FALSE;
  GError *err = NULL;
  gdouble *input_mins;
  gdouble *input_maxs;
  gdouble *scales = NULL;
  gdouble *offsets = NULL;
  gsize gst_input_dims[GST_ONNX_MAX_DIMS];
  GstCaps *mi_caps = NULL;
  gsize i;

  if (parsed_info->inputs->len != 1) {
    GST_ERROR_OBJECT (self, "Only models with 1 input tensor are supported,"
        " but model has %u inputs", parsed_info->inputs->len);
    return FALSE;
  }

  input = &g_array_index (parsed_info->inputs, GstOnnxModelInput, 0);

  if (input->num_dims < 2 || input->num_dims > 4) {
    GST_ERROR_OBJECT (self,
        "Don't know how to interpret tensors with %" G_GSIZE_FORMAT
        " dimensions", input->num_dims);
    return FALSE;
  }

  memcpy (config->dims, input->dims, sizeof (gint64) * input->num_dims);
  memcpy (self->input_dims_model, input->dims, sizeof (self->input_dims_model));
  config->dims_count = input->num_dims;
  config->data_type = input->data_type;

  for (i = 0; i < config->dims_count; i++) {
    if (input->dims[i] < 0)
      gst_input_dims[i] = G_MAXSIZE;
    else
      gst_input_dims[i] = input->dims[i];
  }

  /* Only u8 / f32 inputs are supported right now */
  switch (config->data_type) {
    case GST_TENSOR_DATA_TYPE_UINT8:
    case GST_TENSOR_DATA_TYPE_FLOAT32:
      break;
    default:
      GST_ERROR_OBJECT (self, "Unsupported input tensor data type %s",
          gst_tensor_data_type_get_name (config->data_type));
      return FALSE;
  }

  /* Validation: modelinfo successfully matched dims and datatype from ONNX */
  GST_INFO_OBJECT (self,
      "Input tensor[0]:%s validated - modelinfo matches ONNX model (type: %s)",
      input->onnx_name, gst_tensor_data_type_get_name (config->data_type));

  /* Modelinfo v1.1+ provide the input caps explicitly, only fall back to
   * guessing the tensor layout from its shape when modelinfo doesn't
   * declare a format */
  mi_caps = gst_analytics_modelinfo_get_input_caps (modelinfo,
      input->modelinfo_name);

  /* FIXME: Currently inference element does part of the pre-processing, like
   * scale/offset normalization and buffer layout normalization, for this
   * reason we need to consider this while calculating the sinkpad caps. Once
   * this pre-processing has been removed from the inference element, we
   * should remove the static pad template of the sinkpad and directly use
   * the caps retrieved from modelinfo for the sinkpad. */
  if (mi_caps) {
    GstCaps *tmpl_caps =
        gst_caps_from_string (GST_VIDEO_CAPS_MAKE (GST_ONNX_VIDEO_FORMATS));
    GstCaps *supported_caps = gst_caps_intersect (mi_caps, tmpl_caps);

    gst_caps_unref (tmpl_caps);

    if (!gst_analytics_modelinfo_validate_caps_datatype
        (gst_caps_get_structure (mi_caps, 0), config->data_type)) {
      GST_ERROR_OBJECT (self, "Modelinfo declares caps %" GST_PTR_FORMAT
          " for input tensor '%s', which disagrees with its declared type %s",
          mi_caps, input->modelinfo_name,
          gst_tensor_data_type_get_name (config->data_type));
      gst_caps_unref (supported_caps);
      gst_caps_unref (mi_caps);
      return FALSE;
    }

    if (gst_caps_is_empty (supported_caps)) {
      GST_ERROR_OBJECT (self, "Modelinfo declares caps %" GST_PTR_FORMAT
          " for input tensor '%s', which onnxinference does not support",
          mi_caps, input->modelinfo_name);
      gst_caps_unref (supported_caps);
      gst_caps_unref (mi_caps);
      return FALSE;
    }
    gst_caps_unref (supported_caps);

    /* Resolves the layout from the tensor's declared caps, erroring for a
     * dims/format combination that can't be handled. */
    layout_ok = gst_analytics_video_tensor_layout_from_caps (gst_input_dims,
        config->dims_count, gst_caps_get_structure (mi_caps, 0), &layout, &err);
    format_from_modelinfo = layout_ok &&
        layout.format != GST_VIDEO_FORMAT_UNKNOWN;
  }

  if (layout_ok && !format_from_modelinfo)
    layout_ok = gst_analytics_video_tensor_layout_from_dims (gst_input_dims,
        config->dims_count, &layout, &err);

  if (layout_ok)
    layout_ok = gst_analytics_video_format_guess (layout.format,
        config->data_type, &gst_format, &err);

  if (!layout_ok) {
    GST_ERROR_OBJECT (self, "Can't use input tensor '%s': %s",
        input->modelinfo_name, err->message);
    g_clear_error (&err);
    g_clear_pointer (&mi_caps, gst_caps_unref);
    return FALSE;
  }

  self->height_dim = layout.height_dim;
  self->width_dim = layout.width_dim;
  self->channels_dim = layout.channels_dim;
  self->batch_dim = layout.batch_dim;
  self->planar = layout.planar;

  GST_INFO_OBJECT (self, "Input tensor '%s' format '%s' %s",
      input->modelinfo_name, gst_video_format_to_string (gst_format),
      format_from_modelinfo ? "taken from modelinfo" :
      "guessed from tensor dims");

  self->model_in_height =
      input->dims[self->height_dim] > 0 ? input->dims[self->height_dim] : -1;

  self->model_in_width =
      input->dims[self->width_dim] > 0 ? input->dims[self->width_dim] : -1;

  if (self->channels_dim >= 0)
    self->channels = gst_input_dims[self->channels_dim];
  else
    self->channels = 1;

  GST_DEBUG_OBJECT (self, "height dim[%d]=%d, width dim[%d]=%d,"
      " channels dim[%d]=%d, batch_dim[%d]=%zu planar=%d",
      self->height_dim, self->model_in_height, self->width_dim,
      self->model_in_width, self->channels_dim, self->channels, self->batch_dim,
      self->batch_dim >= 0 ? gst_input_dims[self->batch_dim] : 0, self->planar);

  /* Get per-channel scales and offsets from modelinfo */
  /* For video input, we assume uint8 pixel values in range [0, 255]
   * or f32 in range [0, 1] */
  input_mins = g_alloca (sizeof (gdouble) * self->channels);
  input_maxs = g_alloca (sizeof (gdouble) * self->channels);

  {
    gdouble default_input_max = 255.0;

    switch (config->data_type) {
      case GST_TENSOR_DATA_TYPE_UINT8:
        default_input_max = 255.0;
        break;
      case GST_TENSOR_DATA_TYPE_FLOAT32:
        default_input_max = 1.0;
        break;
      default:
        g_assert_not_reached ();
        return FALSE;
    }

    for (i = 0; i < self->channels; i++) {
      input_mins[i] = 0.0;
      input_maxs[i] = default_input_max;
    }
  }

  if (!gst_analytics_modelinfo_get_input_scales_offsets (modelinfo,
          input->modelinfo_name, self->channels, input_mins, input_maxs,
          NULL, &scales, &offsets)) {
    GST_ERROR_OBJECT (self, "Failed to get scales/offsets for tensor %s",
        input->modelinfo_name);
    g_clear_pointer (&mi_caps, gst_caps_unref);
    return FALSE;
  }

  for (i = 0; i < self->channels; i++) {
    config->scales[i] = scales[i];
    config->offsets[i] = offsets[i];
  }
  g_free (scales);
  g_free (offsets);

  GST_INFO_OBJECT (self, "Input tensor normalization: %u channel(s)",
      self->channels);

  for (i = 0; i < self->channels; i++) {
    GST_DEBUG_OBJECT (self, "  Channel[%" G_GSIZE_FORMAT
        "]: scale=%f, offset=%f", i, config->scales[i], config->offsets[i]);
  }

  /* Setting input tensor caps: when modelinfo declares caps, keep them as
   * the base so any field beyond format
   * is preserved, dims/heuristics only complete fields modelinfo left unset,
   * they never override what modelinfo already pins down. */
  g_mutex_lock (&self->lock);
  gst_caps_unref (self->input_tensors_caps);
  if (mi_caps) {
    self->input_tensors_caps = gst_caps_make_writable (mi_caps);
    mi_caps = NULL;
  } else {
    self->input_tensors_caps = gst_caps_new_empty_simple ("video/x-raw");
  }

  {
    GstStructure *in_caps_s =
        gst_caps_get_structure (self->input_tensors_caps, 0);

    gst_caps_set_simple (self->input_tensors_caps,
        "format", G_TYPE_STRING, gst_video_format_to_string (gst_format), NULL);

    if (!gst_structure_has_field (in_caps_s, "pixel-aspect-ratio"))
      gst_caps_set_simple (self->input_tensors_caps,
          "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL);

    if (self->model_in_width > 0 &&
        !gst_structure_has_field (in_caps_s, "width")) {
      gst_caps_set_simple (self->input_tensors_caps,
          "width", G_TYPE_INT, self->model_in_width, NULL);
    }

    if (self->model_in_height > 0 &&
        !gst_structure_has_field (in_caps_s, "height")) {
      gst_caps_set_simple (self->input_tensors_caps,
          "height", G_TYPE_INT, self->model_in_height, NULL);
    }
  }

  gst_clear_object (&self->importer);
  self->importer = gst_object_ref (importer);
  g_free (self->group_id);
  self->group_id = g_strdup (g_quark_to_string (parsed_info->group_id));

  gst_caps_set_simple (self->output_tensors_caps, "tensors",
      GST_TYPE_STRUCTURE, parsed_info->output_tensorgroups, NULL);
  g_mutex_unlock (&self->lock);

  return TRUE;
}

static GstFlowReturn
gst_onnx_inference_prepare_input_tensors (GstOnnxBaseInference * base,
    GstBuffer * buffer, const GArray * model_inputs, gsize iteration,
    GPtrArray * input_tensors)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (base);
  OrtValue *input = NULL;
  GstOnnxImporterConfig *config = &self->importer_config;

  GST_LOG_OBJECT (self, "Input dimensions: %" G_GINT64_FORMAT
      ":%" G_GINT64_FORMAT ":%" G_GINT64_FORMAT ":%" G_GINT64_FORMAT,
      config->dims_count > 0 ? config->dims[0] : -1,
      config->dims_count > 1 ? config->dims[1] : -1,
      config->dims_count > 2 ? config->dims[2] : -1,
      config->dims_count > 3 ? config->dims[3] : -1);

  if (!self->importer) {
    GST_ERROR_OBJECT (self, "Importer is not prepared");
    return GST_FLOW_ERROR;
  }

  input = gst_onnx_importer_prepare (self->importer, buffer);

  if (!input) {
    GST_ERROR_OBJECT (self, "Couldn't create input tensor");
    return GST_FLOW_ERROR;
  }

  g_ptr_array_add (input_tensors, input);

  return GST_FLOW_OK;
}

static void
gst_onnx_inference_unprepare_input_tensors (GstOnnxBaseInference * base,
    GstBuffer * buffer, GPtrArray * input_tensors)
{
  GstOnnxInference *self = GST_ONNX_INFERENCE (base);

  if (self->importer)
    gst_onnx_importer_unprepare (self->importer);
}
