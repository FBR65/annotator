"""Kriterien A1 bis A3 (Geometrie) und iou."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from annotator.geometry import Box, BoxError, clamp_box, from_yolo_line, iou, to_yolo_line

DEC6 = st.floats(min_value=0.000002, max_value=0.998, allow_nan=False, allow_infinity=False)


def test_a1_line_exact_text() -> None:
    box = Box(1, 0.123456, 0.654321, 0.2, 0.3)
    assert to_yolo_line(box) == "1 0.123456 0.654321 0.200000 0.300000"


def test_a1_roundtrip_exact() -> None:
    box = Box(2, 0.123456, 0.654321, 0.2, 0.3)
    assert from_yolo_line(to_yolo_line(box)) == box


@settings(max_examples=500, deadline=None)
@given(cx=DEC6, cy=DEC6, w=DEC6, h=DEC6, cls=st.integers(min_value=0, max_value=999))
def test_a1_property_roundtrip(cx: float, cy: float, w: float, h: float, cls: int) -> None:
    box = Box(cls, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
    again = from_yolo_line(to_yolo_line(box))
    assert abs(again.cx - box.cx) <= 1e-6
    assert abs(again.cy - box.cy) <= 1e-6
    assert abs(again.w - box.w) <= 1e-6
    assert abs(again.h - box.h) <= 1e-6
    assert again.class_id == box.class_id
    for value in (again.cx, again.cy, again.w, again.h):
        assert 0.0 <= value <= 1.0


def test_a1_bad_line_rejected() -> None:
    with pytest.raises(BoxError):
        from_yolo_line("1 0.5 0.5")


def test_a2_clip_to_border() -> None:
    out = clamp_box(Box(0, 0.95, 0.5, 0.4, 0.2), width=100, height=50)
    assert out.cx + out.w / 2 <= 1.0 + 1e-9
    assert out.w > 0
    assert out.h > 0


def test_a2_min_width_one_pixel() -> None:
    out = clamp_box(Box(0, 0.5, 0.5, 0.0000001, 0.5), width=100, height=50)
    assert out.w * 100 >= 1.0 - 1e-9
    assert out.cx - out.w / 2 >= -1e-9
    assert out.cx + out.w / 2 <= 1.0 + 1e-9


def test_a2_outside_fully_lands_inside() -> None:
    out = clamp_box(Box(0, 1.4, 0.5, 0.2, 0.2), width=100, height=50)
    assert out.cx - out.w / 2 >= 0.0
    assert out.cx + out.w / 2 <= 1.0 + 1e-9


def test_a3_zero_width_rejected_with_file_name() -> None:
    with pytest.raises(BoxError) as exc:
        from annotator.geometry import validate_box

        validate_box(Box(0, 0.5, 0.5, 0.0, 0.2), "bild_007.png")
    assert "bild_007.png" in str(exc.value)


def test_a3_negative_class_rejected() -> None:
    from annotator.geometry import validate_box

    with pytest.raises(BoxError):
        validate_box(Box(-1, 0.5, 0.5, 0.2, 0.2), "x.png")


def test_iou_identical_and_disjoint() -> None:
    box = Box(0, 0.5, 0.5, 0.2, 0.2)
    assert iou(box, box) == pytest.approx(1.0)
    far = Box(0, 0.05, 0.05, 0.02, 0.02)
    assert iou(box, far) == pytest.approx(0.0)
