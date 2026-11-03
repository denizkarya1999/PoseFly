"""Detected drone boxes and their associated feature estimates."""
import cv2

# Older results remain visible, but are marked as last-known positions.
FRESH_BOX_AGE = 0.5
FEATURE_LABELS = {'angle': 'Angle', 'distance': 'Distance', 'led': 'LED ID', 'speed': 'Speed'}


def feature_values(features):
    """Keep every feature present, even before inference or when it is disabled."""
    values = []
    for name in FEATURE_LABELS:
        label, confidence = (features or {}).get(name, (None, None))
        value = f'{label} ({confidence:.0%})' if label is not None and confidence is not None else 'N/A'
        values.append(value)
    return values


def feature_lines(features):
    return [f'{title}: {value}' for title, value in zip(FEATURE_LABELS.values(), feature_values(features))]


def draw_drones(frame, detections, features=None, age=None):
    result = frame.copy()
    h, w = result.shape[:2]
    features = features or []
    color = (0, 180, 255) if age is not None else (0, 220, 0)
    outlines = []
    for index, (x1, y1, x2, y2, confidence) in enumerate(detections):
        x1, x2 = max(0, min(w - 1, int(x1))), max(0, min(w - 1, int(x2)))
        y1, y2 = max(0, min(h - 1, int(y1))), max(0, min(h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        outlines.append((x1, y1, x2, y2))
        age_label = f' [{age:.1f}s ago]' if age is not None else ''
        lines = [f'Drone {confidence:.0%}{age_label}']
        lines.extend(feature_lines(features[index] if index < len(features) else {}))
        # Size the complete label to the image, not to a potentially tiny drone box.
        font = cv2.FONT_HERSHEY_SIMPLEX
        widths = [cv2.getTextSize(line, font, .55, 1)[0][0] for line in lines]
        scale = .55 * min(1., max(1, w - 8) / max(1, max(widths)), max(1, h - 8) / (22 * len(lines)))
        line_height = max(1, round(22 * scale / .55))
        text_width = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines)
        panel_height = line_height * len(lines) + 6
        left = max(0, min(x1, w - text_width - 8))
        top = y1 - panel_height if y1 >= panel_height else min(y1 + 2, max(0, h - panel_height))
        cv2.rectangle(result, (left, top), (min(w - 1, left + text_width + 8), min(h - 1, top + panel_height)), (0, 0, 0), -1)
        for row, line in enumerate(lines):
            cv2.putText(result, line, (left + 4, top + (row + 1) * line_height),
                        font, scale, color, 1, cv2.LINE_AA)
    # Labels can cover small or overlapping drones. Paint every outline last.
    for x1, y1, x2, y2 in outlines:
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
    return result


def draw_recent_drones(frame, detections, captured_at, source_shape, now, features=None):
    if captured_at is None or source_shape != frame.shape[:2] or not detections:
        return frame
    age = max(0., now - captured_at)
    # Do not discard a valid detection just because inference was slow. The next
    # detector result replaces these boxes (including an empty result).
    return draw_drones(frame, detections, features, age=age if age > FRESH_BOX_AGE else None)
