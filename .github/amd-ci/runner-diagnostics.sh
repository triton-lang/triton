#!/usr/bin/env bash

set +e
shopt -s nullglob

run() {
  echo "+ $*"
  "$@" 2>&1 || true
}

echo "CI_CONTAINER_IMAGE=$CI_CONTAINER_IMAGE"
echo "CI_RUNNER_LABELS=$CI_RUNNER_LABELS"
for name in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION; do
  echo "$name=${!name-<unset>}"
done
run uname -a
run cat /proc/cmdline
run id
run ls -ldn /dev/kfd /dev/dri
render_nodes=(/dev/dri/renderD*)
echo "DRM render nodes: ${#render_nodes[@]}"
stat -c 'render node permissions: mode=%a uid=%u gid=%g' "${render_nodes[@]}" 2>/dev/null | sort -u
run cat /sys/module/amdgpu/version
for parameter in sched_policy lockup_timeout debug_mask ip_block_mask; do
  run cat "/sys/module/amdgpu/parameters/$parameter"
done

for attribute in product_name product_number vbios_version; do
  echo "$attribute:"
  grep -h . /sys/class/drm/card*/device/"$attribute" 2>/dev/null | sort -u
done
echo "KFD topology:"
for node in /sys/class/kfd/kfd/topology/nodes/*; do
  [[ -r "$node/gpu_id" && "$(<"$node/gpu_id")" != 0 ]] || continue
  grep -E '^(vendor_id|device_id|simd_count|max_waves_per_simd|fw_version) ' "$node/properties"
done | sort -u

if command -v rocprofv3-avail >/dev/null 2>&1; then
  pc_sampling_output=$(ROCPROFILER_LOG_LEVEL=info rocprofv3-avail info --pc-sampling 2>&1)
  status=$?
  echo "rocprofv3-avail exit status: $status"
  if ((status == 0)); then
    echo "PC-sampling agents: $(grep -c '^GPU' <<<"$pc_sampling_output")"
    grep -Ei 'PC sampling unavailable|failed|incompatible|not support|Method|Unit|Min_Interval|Max_Interval|Flags' \
      <<<"$pc_sampling_output" | sort -u
  else
    echo "$pc_sampling_output"
  fi
else
  echo "rocprofv3-avail: unavailable"
fi

exit 0
