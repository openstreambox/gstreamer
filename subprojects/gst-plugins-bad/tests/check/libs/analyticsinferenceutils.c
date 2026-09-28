/* GStreamer
 * Copyright (C) 2026 Collabora Ltd
 * @author: Daniel Morin <daniel.morin@collabora.com>
 *
 * analyticsinferenceutils.c
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

/* A fictitious accelerator memory feature. */
#define ACCEL_MEMORY_FEATURE "memory:TestAccel"

static GstCaps *
new_any_features_caps (const gchar * media_type)
{
  GstCaps *caps = gst_caps_new_empty ();

  gst_caps_append_structure_full (caps, gst_structure_new_empty (media_type),
      gst_caps_features_new_any ());

  return caps;
}

static GstStructure *
new_tensorgroups (const gchar * group_id, gint value)
{
  return gst_structure_new ("tensorgroups", group_id, G_TYPE_INT, value, NULL);
}

static void
set_tensorgroups (GstCaps * caps, guint idx, GstStructure * groups)
{
  gst_structure_set (gst_caps_get_structure (caps, idx), "tensors",
      GST_TYPE_STRUCTURE, groups, NULL);
  gst_structure_free (groups);
}

/* gst_analytics_inference_transform_sink_caps() */
GST_START_TEST (test_sink_restricts_against_restrictions)
{
  /* Verify sink caps are restricted against the model input restrictions */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]");
  GstCaps *result;
  const GValue *width;

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, NULL, NULL);

  fail_unless_equals_int (gst_caps_get_size (result), 1);
  width = gst_structure_get_value (gst_caps_get_structure (result, 0), "width");
  fail_unless (GST_VALUE_HOLDS_INT_RANGE (width));
  fail_unless_equals_int (gst_value_get_int_range_min (width), 1);
  fail_unless_equals_int (gst_value_get_int_range_max (width), 1920);

  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_sink_keeps_restrictions_caps_features)
{
  /* Verify caps features the caller adds to the restrictions, like an
   * memory for accelerator's, survive the transform */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]; "
      "video/x-raw(" ACCEL_MEMORY_FEATURE "), width=(int)[1,1920]");
  GstCaps *caps = new_any_features_caps ("video/x-raw");
  GstCaps *result;
  gboolean found_accel_structure = FALSE;
  guint i;

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, NULL, NULL);

  fail_unless_equals_int (gst_caps_get_size (result), 2);
  for (i = 0; i < gst_caps_get_size (result); i++) {
    if (gst_caps_features_contains (gst_caps_get_features (result, i),
            ACCEL_MEMORY_FEATURE))
      found_accel_structure = TRUE;
  }
  fail_unless (found_accel_structure);

  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_sink_merges_own_group_with_incoming_group)
{
  /* Verify our own tensor group is added alongside a group declared by
   * another inference element earlier in the pipeline, not replacing it */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstStructure *model_groups = new_tensorgroups ("yolo", 1);
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]");
  GstCaps *result;
  const GValue *tensors_value;
  const GstStructure *result_groups;

  set_tensorgroups (caps, 0, new_tensorgroups ("face-detector", 2));

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, model_groups, "yolo");

  fail_unless_equals_int (gst_caps_get_size (result), 1);
  tensors_value = gst_structure_get_value (gst_caps_get_structure (result, 0),
      "tensors");
  fail_unless (tensors_value != NULL);
  result_groups = gst_value_get_structure (tensors_value);
  fail_unless (gst_structure_has_field (result_groups, "face-detector"));
  fail_unless (gst_structure_has_field (result_groups, "yolo"));

  gst_structure_free (model_groups);
  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_sink_without_group_id_keeps_incoming_groups)
{
  /* Verify an element declaring no tensor group passes the incoming groups
   * through unchanged */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]");
  GstCaps *result;
  const GValue *tensors_value;
  const GstStructure *result_groups;

  set_tensorgroups (caps, 0, new_tensorgroups ("face-detector", 2));

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, NULL, NULL);

  fail_unless_equals_int (gst_caps_get_size (result), 1);
  tensors_value = gst_structure_get_value (gst_caps_get_structure (result, 0),
      "tensors");
  fail_unless (tensors_value != NULL);
  result_groups = gst_value_get_structure (tensors_value);
  fail_unless_equals_int (gst_structure_n_fields (result_groups), 1);
  fail_unless (gst_structure_has_field (result_groups, "face-detector"));

  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_sink_drops_structure_on_own_group_upstream)
{
  /* Verify a structure is dropped when it already declares our own group_id */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstStructure *model_groups = new_tensorgroups ("yolo", 1);
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]");
  GstCaps *result;

  set_tensorgroups (caps, 0, new_tensorgroups ("yolo", 99));

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, model_groups, "yolo");

  fail_unless (gst_caps_is_empty (result));

  gst_structure_free (model_groups);
  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_sink_drops_structure_when_model_has_no_group)
{
  /* Verify a structure is dropped when the model declares no tensor group
   * for group_id */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstStructure *model_groups = new_tensorgroups ("ssd", 1);
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]");
  GstCaps *result;

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, NULL, "yolo");
  fail_unless (gst_caps_is_empty (result));
  gst_caps_unref (result);

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, model_groups, "yolo");
  fail_unless (gst_caps_is_empty (result));
  gst_caps_unref (result);

  gst_structure_free (model_groups);
  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_sink_any_caps_uses_restrictions)
{
  /* Verify ANY sink caps give the restrictions with our own group added */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]; "
      "video/x-raw(" ACCEL_MEMORY_FEATURE "), width=(int)[1,1920]");
  GstStructure *model_groups = new_tensorgroups ("yolo", 1);
  GstCaps *caps = gst_caps_new_any ();
  GstCaps *result;
  guint i;

  result = gst_analytics_inference_transform_sink_caps (caps,
      restrictions, model_groups, "yolo");

  fail_unless_equals_int (gst_caps_get_size (result), 2);
  for (i = 0; i < gst_caps_get_size (result); i++) {
    GstStructure *s = gst_caps_get_structure (result, i);
    const GValue *tensors_value = gst_structure_get_value (s, "tensors");

    fail_unless (tensors_value != NULL);
    fail_unless (gst_structure_has_field (gst_value_get_structure
            (tensors_value), "yolo"));
    gst_structure_remove_field (s, "tensors");
  }
  fail_unless (gst_caps_is_equal (result, restrictions));
  gst_caps_unref (result);

  /* Without restrictions there is nothing to add the group to */
  result = gst_analytics_inference_transform_sink_caps (caps, caps,
      model_groups, "yolo");
  fail_unless (gst_caps_is_any (result));
  gst_caps_unref (result);

  gst_structure_free (model_groups);
  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
}

GST_END_TEST;

/* gst_analytics_inference_transform_src_caps() */
GST_START_TEST (test_src_restricts_against_restrictions)
{
  /* Verify src caps are restricted against the model input restrictions */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]");
  GstCaps *result;
  const GValue *width;

  result = gst_analytics_inference_transform_src_caps (caps, restrictions);

  fail_unless_equals_int (gst_caps_get_size (result), 1);
  width = gst_structure_get_value (gst_caps_get_structure (result, 0), "width");
  fail_unless (GST_VALUE_HOLDS_INT_RANGE (width));
  fail_unless_equals_int (gst_value_get_int_range_min (width), 1);
  fail_unless_equals_int (gst_value_get_int_range_max (width), 1920);

  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_src_drops_downstream_tensor_groups)
{
  /* Verify tensor groups proposed by downstream are not propagated
   * upstream, whether or not they are our own group */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]");
  GstCaps *caps = gst_caps_from_string ("video/x-raw, width=(int)[1,4096]; "
      "video/x-raw, width=(int)[1,4096]");
  GstCaps *result;
  guint i;

  set_tensorgroups (caps, 0, new_tensorgroups ("ssd", 2));
  set_tensorgroups (caps, 1, new_tensorgroups ("yolo", 1));

  result = gst_analytics_inference_transform_src_caps (caps, restrictions);

  fail_if (gst_caps_is_empty (result));
  for (i = 0; i < gst_caps_get_size (result); i++)
    fail_if (gst_structure_has_field (gst_caps_get_structure (result, i),
            "tensors"));

  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

GST_START_TEST (test_src_any_caps_returns_restrictions)
{
  /* Verify ANY src caps give the restrictions */
  GstCaps *restrictions =
      gst_caps_from_string ("video/x-raw, width=(int)[1,1920]; "
      "video/x-raw(" ACCEL_MEMORY_FEATURE "), width=(int)[1,1920]");
  GstCaps *caps = gst_caps_new_any ();
  GstCaps *result;

  result = gst_analytics_inference_transform_src_caps (caps, restrictions);

  fail_unless (gst_caps_is_equal (result, restrictions));

  gst_caps_unref (restrictions);
  gst_caps_unref (caps);
  gst_caps_unref (result);
}

GST_END_TEST;

/* gst_analytics_inference_caps_has_tensor_group() */
GST_START_TEST (test_caps_has_tensor_group_true_when_present)
{
  /* Verify a declared group is found and other names are rejected */
  GstStructure *groups = gst_structure_new ("tensorgroups",
      "yolo", G_TYPE_INT, 1, NULL);
  GstCaps *caps = gst_caps_from_string ("video/x-raw");

  gst_structure_set (gst_caps_get_structure (caps, 0), "tensors",
      GST_TYPE_STRUCTURE, groups, NULL);
  gst_structure_free (groups);

  fail_unless (gst_analytics_inference_caps_has_tensor_group (caps, "yolo"));
  fail_if (gst_analytics_inference_caps_has_tensor_group (caps, "ssd"));
  fail_if (gst_analytics_inference_caps_has_tensor_group (caps, NULL));

  gst_caps_unref (caps);
}

GST_END_TEST;

GST_START_TEST (test_caps_has_tensor_group_false_without_tensors_field)
{
  /* Verify a caps with no tensors field at all reports no group */
  GstCaps *caps = gst_caps_from_string ("video/x-raw");

  fail_if (gst_analytics_inference_caps_has_tensor_group (caps, "yolo"));

  gst_caps_unref (caps);
}

GST_END_TEST;

static Suite *
analyticsinferenceutils_suite (void)
{
  Suite *s = suite_create ("GstAnalyticsInferenceUtils");
  TCase *tc_sink = tcase_create ("inference-sink-caps");
  TCase *tc_src = tcase_create ("inference-src-caps");
  TCase *tc_has_tensor_group = tcase_create ("caps-has-tensor-group");

  suite_add_tcase (s, tc_sink);
  tcase_add_test (tc_sink, test_sink_restricts_against_restrictions);
  tcase_add_test (tc_sink, test_sink_keeps_restrictions_caps_features);
  tcase_add_test (tc_sink, test_sink_merges_own_group_with_incoming_group);
  tcase_add_test (tc_sink, test_sink_without_group_id_keeps_incoming_groups);
  tcase_add_test (tc_sink, test_sink_drops_structure_on_own_group_upstream);
  tcase_add_test (tc_sink, test_sink_drops_structure_when_model_has_no_group);
  tcase_add_test (tc_sink, test_sink_any_caps_uses_restrictions);

  suite_add_tcase (s, tc_src);
  tcase_add_test (tc_src, test_src_restricts_against_restrictions);
  tcase_add_test (tc_src, test_src_drops_downstream_tensor_groups);
  tcase_add_test (tc_src, test_src_any_caps_returns_restrictions);

  suite_add_tcase (s, tc_has_tensor_group);
  tcase_add_test (tc_has_tensor_group,
      test_caps_has_tensor_group_true_when_present);
  tcase_add_test (tc_has_tensor_group,
      test_caps_has_tensor_group_false_without_tensors_field);

  return s;
}

GST_CHECK_MAIN (analyticsinferenceutils);
