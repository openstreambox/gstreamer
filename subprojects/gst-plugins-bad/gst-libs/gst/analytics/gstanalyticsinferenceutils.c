/* GStreamer
 * Copyright (C) 2026 Collabora Ltd
 * @author: Daniel Morin <daniel.morin@collabora.com>
 *
 * gstanalyticsinferenceutils.c
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

#include "gstanalyticsinferenceutils.h"

/**
 * gst_analytics_inference_caps_has_tensor_group:
 * @caps: (transfer none): caps to check
 * @group_id: (nullable): tensor group id to look for
 *
 * Checks whether any structure in @caps declares a tensor group named
 * @group_id in its `tensors` field.
 *
 * Returns: %TRUE if @caps declares a tensor group named @group_id, %FALSE
 *    otherwise or if @group_id is %NULL
 *
 * Since: 1.30
 */
gboolean
gst_analytics_inference_caps_has_tensor_group (const GstCaps * caps,
    const gchar * group_id)
{
  guint i;

  g_return_val_if_fail (caps != NULL, FALSE);

  if (!group_id)
    return FALSE;

  for (i = 0; i < gst_caps_get_size (caps); i++) {
    const GstStructure *s = gst_caps_get_structure (caps, i);
    const GValue *tensors_value = gst_structure_get_value (s, "tensors");
    const GstStructure *tensorgroups = tensors_value ?
        gst_value_get_structure (tensors_value) : NULL;

    if (tensorgroups && gst_structure_has_field (tensorgroups, group_id))
      return TRUE;
  }

  return FALSE;
}

/* Add the tensor group named group_id from model_groups into dest_groups.
 * Returns FALSE without modifying dest_groups if model_groups doesn't have
 * that group, or if dest_groups already has a group with that same id. */
static gboolean
merge_tensor_group (GstStructure * dest_groups,
    const GstStructure * model_groups, const gchar * group_id)
{
  const GValue *group_value;

  if (gst_structure_has_field (dest_groups, group_id))
    return FALSE;

  group_value = gst_structure_get_value (model_groups, group_id);
  if (!group_value)
    return FALSE;

  gst_structure_set_value (dest_groups, group_id, group_value);

  return TRUE;
}

/**
 * gst_analytics_inference_transform_sink_caps:
 * @caps: (transfer none): caps on the sink pad
 * @restrictions: (transfer none): caps the model accepts as input, including
 *    any caps features supported by the caller
 * @model_tensorgroups: (transfer none) (nullable): the tensor groups the model
 *    produces, keyed by group id, as found in the `tensors` field of the
 *    caps describing the model output
 * @group_id: (nullable): id of the tensor group the model produces, as
 *    declared in its model info, or %NULL if the model doesn't declare one
 *    for its output
 *
 * Transforms @caps, the caps on the sink pad, into the caps for the src pad.
 *
 * Each structure of @caps, without its `tensors` field, is intersected with
 * @restrictions. The tensor groups already declared in @caps are kept and the
 * group named @group_id from @model_tensorgroups is added to them.
 *
 * A structure is dropped if @group_id is set and either @caps already declares
 * a group with that id, which happens when the same model is chained twice, or
 * @model_tensorgroups has no group with that id.
 *
 * If @caps is ANY, @restrictions is used in its place, so the result is
 * @restrictions with the group named @group_id added. If both @caps and
 * @restrictions are ANY, ANY is returned.
 *
 * Returns: (transfer full): the src caps
 *
 * Since: 1.30
 */
GstCaps *
gst_analytics_inference_transform_sink_caps (GstCaps * caps,
    GstCaps * restrictions, const GstStructure * model_tensorgroups,
    const gchar * group_id)
{
  GstCaps *other_caps;
  guint i;

  g_return_val_if_fail (caps != NULL, NULL);
  g_return_val_if_fail (restrictions != NULL, NULL);

  if (gst_caps_is_any (caps)) {
    if (gst_caps_is_any (restrictions))
      return gst_caps_new_any ();
    caps = restrictions;
  }

  other_caps = gst_caps_new_empty ();

  for (i = 0; i < gst_caps_get_size (caps); i++) {
    const GstStructure *input_s = gst_caps_get_structure (caps, i);
    const GstCapsFeatures *features = gst_caps_get_features (caps, i);
    const GValue *tensors_value = gst_structure_get_value (input_s, "tensors");
    const GstStructure *incoming_tensorgroups = tensors_value ?
        gst_value_get_structure (tensors_value) : NULL;
    GstStructure *merged_tensorgroups = incoming_tensorgroups ?
        gst_structure_copy (incoming_tensorgroups) :
        gst_structure_new_empty ("tensorgroups");
    GstStructure *candidate_s = gst_structure_copy (input_s);
    GstCaps *candidate = gst_caps_new_empty ();
    GstCaps *intersect;

    gst_structure_remove_field (candidate_s, "tensors");
    gst_caps_append_structure_full (candidate, candidate_s,
        gst_caps_features_copy (features));
    intersect = gst_caps_intersect_full (candidate, restrictions,
        GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (candidate);

    if (gst_caps_is_empty (intersect)) {
      gst_caps_unref (intersect);
      gst_structure_free (merged_tensorgroups);
      continue;
    }

    if (group_id &&
        (!model_tensorgroups ||
            !merge_tensor_group (merged_tensorgroups,
                model_tensorgroups, group_id))) {
      GST_WARNING ("Tensor group '%s' is already present in caps or missing "
          "from the model tensor groups, dropping caps structure", group_id);
      gst_caps_unref (intersect);
      gst_structure_free (merged_tensorgroups);
      continue;
    }

    if (gst_structure_n_fields (merged_tensorgroups) > 0) {
      guint j;

      intersect = gst_caps_make_writable (intersect);
      for (j = 0; j < gst_caps_get_size (intersect); j++) {
        gst_structure_set (gst_caps_get_structure (intersect, j),
            "tensors", GST_TYPE_STRUCTURE, merged_tensorgroups, NULL);
      }
    }

    gst_structure_free (merged_tensorgroups);
    gst_caps_append (other_caps, intersect);
  }

  return other_caps;
}

/**
 * gst_analytics_inference_transform_src_caps:
 * @caps: (transfer none): caps on the src pad
 * @restrictions: (transfer none): caps the model accepts as input, including
 *    any caps features supported by the caller
 *
 * Transforms @caps, the caps on the src pad, into the caps for the sink pad.
 *
 * The `tensors` field is removed from every structure of @caps, so the tensor
 * groups downstream expects are not propagated upstream, then the result is
 * intersected with @restrictions. This is the counterpart of
 * gst_analytics_inference_transform_sink_caps(), which adds the tensor group
 * back on the way down.
 *
 * If @caps is ANY, the result is @restrictions.
 *
 * Returns: (transfer full): the sink caps
 *
 * Since: 1.30
 */
GstCaps *
gst_analytics_inference_transform_src_caps (GstCaps * caps,
    GstCaps * restrictions)
{
  GstCaps *tmp_caps;
  GstCaps *other_caps;
  guint i;

  g_return_val_if_fail (caps != NULL, NULL);
  g_return_val_if_fail (restrictions != NULL, NULL);

  tmp_caps = gst_caps_copy (caps);
  for (i = 0; i < gst_caps_get_size (tmp_caps); i++)
    gst_structure_remove_field (gst_caps_get_structure (tmp_caps, i),
        "tensors");

  other_caps = gst_caps_intersect_full (tmp_caps, restrictions,
      GST_CAPS_INTERSECT_FIRST);
  gst_caps_unref (tmp_caps);

  return other_caps;
}
