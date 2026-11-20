import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chili_harvest import vision_utils as vu  # noqa: E402


def test_labels():
    assert vu.normalize_label('Semi-Ripe') == 'semiripe'
    assert vu.normalize_label('unripe') != vu.normalize_label('ripe')


def test_match_chili():
    assert vu.match_chili([10, 10, 50, 100], [[200, 200, 240, 300], [12, 12, 52, 98]]) == 1
    assert vu.match_chili([10, 10, 50, 100], [[200, 200, 240, 300]]) is None


def _scene():
    # chili hanging at x 40..60, y 50..120; its stem above at x 48..52, y 20..50
    # a second stem far away at x 150..154
    h, w = 200, 200
    chili = np.zeros((h, w), bool)
    chili[50:120, 40:60] = True
    stem = np.zeros((h, w), bool)
    stem[20:50, 48:52] = True
    other = np.zeros((h, w), bool)
    other[20:50, 150:154] = True
    return chili, stem, other


def test_select_and_cut():
    chili, stem, other = _scene()
    idx, attach = vu.select_stem([other, stem], chili_mask=chili, gap_px=5)
    assert idx == 1
    u, v = vu.cut_pixel(stem, attach, ratio=0.5)
    assert 48 <= u < 52 and 30 <= v <= 40   # middle of the stem
    u0, v0 = vu.cut_pixel(stem, attach, ratio=0.0)
    assert v0 >= 44                          # near the chili
    idx, _ = vu.select_stem([other], chili_mask=chili, gap_px=5)
    assert idx is None                       # far stem is rejected


def test_depth():
    chili, stem, _ = _scene()
    depth = np.zeros(chili.shape, np.uint16)
    depth[chili] = 400
    depth[stem] = 900                         # stem "missed", sees background
    zs = vu.robust_depth_mm(depth, stem)
    zc = vu.robust_depth_mm(depth, chili)
    z, src = vu.fuse_depth(zs, zc)
    assert (z, src) == (400.0, 'chili')
    depth[stem] = 410
    z, src = vu.fuse_depth(vu.robust_depth_mm(depth, stem), zc)
    assert (z, src) == (410.0, 'stem')


def test_deproject():
    K = [[500, 0, 320], [0, 500, 240], [0, 0, 1]]
    x, y, z = vu.deproject(320 + 50, 240, 0.5, K)
    assert abs(x - 0.05) < 1e-9 and y == 0 and z == 0.5
