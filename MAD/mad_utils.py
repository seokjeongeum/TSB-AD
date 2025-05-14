import numpy as np
import re

PLOT_FIG_WIDTH = 400
PLOT_NBINS = 600


def _find_clusters(indices):
    if not indices:  # Simplified check
        return []
    indices = np.unique(np.asarray(indices, dtype=int))
    if len(indices) == 0:
        return []
    if len(indices) == 1:
        return [(indices[0], indices[0])]

    diffs = np.diff(indices)
    split_points = np.where(diffs > 1)[0]
    starts = np.insert(indices[split_points + 1], 0, indices[0])
    ends = np.append(indices[split_points], indices[-1])
    return list(zip(starts, ends))


def _resolve_overlapping_ranges(raw_ranges, n_samples):
    if not raw_ranges:
        return []

    points = set()
    for start, end in raw_ranges:
        # Basic check to prevent errors if start/end are not int
        # This is a minimal guard.
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start <= end < n_samples
        ):
            points.add(start)
            points.add(end + 1)

    if not points:  # If no valid ranges were processed
        return []

    sorted_points = sorted(list(points))

    unique_sorted_points = []
    if sorted_points:
        unique_sorted_points.append(sorted_points[0])
        for i in range(1, len(sorted_points)):
            if sorted_points[i] > sorted_points[i - 1]:
                unique_sorted_points.append(sorted_points[i])

    P = [p for p in unique_sorted_points if 0 <= p <= n_samples]
    if not P:
        return []
    P = sorted(list(set(P)))

    resolved = []
    if len(P) < 2:
        if len(P) == 1 and P[0] < n_samples:
            single_pt = P[0]
            for (
                r_start,
                r_end,
            ) in raw_ranges:
                if (
                    isinstance(r_start, int)
                    and isinstance(r_end, int)
                    and r_start == single_pt
                    and r_end == single_pt
                ):
                    resolved.append([single_pt, single_pt])
                    break
        return resolved

    for i in range(len(P) - 1):
        s_current = P[i]
        e_current = P[i + 1] - 1
        if s_current > e_current or s_current >= n_samples or e_current < 0:
            continue

        current_segment = [s_current, e_current]
        mid_point = (s_current + e_current) / 2.0
        is_covered = False
        for r_start, r_end in raw_ranges:
            if (
                isinstance(r_start, int)
                and isinstance(r_end, int)
                and r_start <= mid_point <= r_end
            ):
                is_covered = True
                break
        if is_covered:
            resolved.append(current_segment)
    return resolved


def _merge_contiguous_ranges(ranges_list):
    if not ranges_list:
        return []
    merged = []
    # Assuming ranges_list is sorted by start index
    for current_start, current_end in ranges_list:
        if not merged or current_start > merged[-1][1] + 1:
            merged.append([current_start, current_end])
        else:
            merged[-1][1] = max(merged[-1][1], current_end)
    return merged


def extract_base_dataset_name(filename):
    if not filename:
        return "unknown_dataset"
    parts = filename.split("_")
    if len(parts) > 1:
        known_prefixes = ["MSL", "SMAP", "SMD", "NAB", "UCR", "MBA", "ECG", "YAHOO"]
        for part in parts:
            if part.upper() in known_prefixes:
                return part.upper()
        potential_name = parts[1] if len(parts) > 1 else "unknown"
        if potential_name.isupper() or (potential_name and potential_name[0].isupper()):
            return potential_name
        return "unknown_dataset"
    return "unknown_dataset"


def extract_domain_from_filename(filename):
    if not filename:
        return "unknown_domain"
    match = re.search(r"_id_.*?_(.*?)_tr_", filename)
    if match:
        return match.group(1)
    parts = filename.split("_")
    if len(parts) > 3 and parts[2].lower() == "id":
        return parts[3]
    elif len(parts) > 1:
        return parts[1]
    return "unknown_domain"


def strip_markdown_code_fences(code_string):
    # Assuming code_string is a string
    pattern_python = r"^\s*```python\n(.*?)\n```\s*$"
    match = re.match(pattern_python, code_string, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    pattern_generic = r"^\s*```\n(.*?)\n```\s*$"
    match = re.match(pattern_generic, code_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    code = re.sub(r"^\s*```[a-zA-Z]*\n?", "", code_string)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip() 