"""Pure numpy / OpenCV helpers for the chili perception node.

Kept free of ROS imports so they can be unit-tested on any PC:
    python3 -m pytest test/test_vision_utils.py
"""
import re

import cv2
import numpy as np


def normalize_label(name):
    """'Semi-Ripe' -> 'semiripe', 'ripe ' -> 'ripe'. Exact match after this
    avoids the classic bug where 'unripe' contains 'ripe'."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def box_iou(a, b):
    """IoU of two boxes [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_chili(ripe_box, chili_boxes, min_iou=0.3):
    """Index of the segmentation 'chili' box that best overlaps a ripeness
    box, or None."""
    best, best_iou = None, min_iou
    for i, cb in enumerate(chili_boxes):
        iou = box_iou(ripe_box, cb)
        if iou >= best_iou:
            best, best_iou = i, iou
    return best


def _box_mask(shape, box):
    m = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
    return m


def select_stem(stem_masks, chili_mask=None, chili_box=None, gap_px=12):
    """Pick the stem attached to one chili.

    Score = number of stem pixels inside the chili region grown by gap_px.
    Uses the chili mask when available, otherwise the ripeness box.
    Returns (index, attach_mask) or (None, None).
    """
    if not stem_masks:
        return None, None
    shape = stem_masks[0].shape
    if chili_mask is not None and chili_mask.any():
        region = chili_mask
    elif chili_box is not None:
        region = _box_mask(shape, chili_box)
    else:
        return None, None
    k = max(1, int(gap_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    grown = cv2.dilate(region.astype(np.uint8), kernel).astype(bool)

    best, best_score, best_attach = None, 0, None
    for i, sm in enumerate(stem_masks):
        attach = sm & grown
        score = int(attach.sum())
        if score > best_score:
            best, best_score, best_attach = i, score, attach
    return best, best_attach


def cut_pixel(stem_mask, attach_mask, ratio=0.5):
    """Point on the stem, `ratio` of the way from where it joins the chili
    (0.0) to its far end (1.0). Snapped to a real stem pixel. Returns (u, v)."""
    ys, xs = np.nonzero(stem_mask)
    if len(xs) == 0:
        return None
    pts = np.stack([xs, ys], axis=1).astype(float)
    if attach_mask is not None and attach_mask.any():
        ay, ax = np.nonzero(attach_mask)
        a = np.array([ax.mean(), ay.mean()])
    else:
        a = pts.mean(axis=0)
    far = pts[np.argmax(np.linalg.norm(pts - a, axis=1))]
    goal = a + float(np.clip(ratio, 0.0, 1.0)) * (far - a)
    nearest = pts[np.argmin(np.linalg.norm(pts - goal, axis=1))]
    return int(nearest[0]), int(nearest[1])


def robust_depth_mm(depth, mask, min_mm=150, max_mm=1500, min_pixels=5):
    """Median depth (mm) of valid pixels under a mask, or None."""
    if depth is None or mask is None:
        return None
    vals = depth[mask]
    vals = vals[(vals > min_mm) & (vals < max_mm)]
    if vals.size < min_pixels:
        return None
    return float(np.median(vals))


def fuse_depth(z_stem, z_chili, max_jump_mm=30.0):
    """Stem depth if it agrees with the chili body, else the chili depth.
    Returns (z_mm, source) or (None, None)."""
    if z_stem is not None and (z_chili is None or abs(z_stem - z_chili) <= max_jump_mm):
        return z_stem, 'stem'
    if z_chili is not None:
        return z_chili, 'chili'
    return None, None


def deproject(u, v, z_m, K):
    """Pixel + depth (m) -> 3D point (m) in the optical frame of camera K."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    return ((u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m)
