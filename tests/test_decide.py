"""Kriterien C1 bis C6 und E7: Entscheidungslogik und Start-Hinweis."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from annotator.decide import DecideConfig, Detection, decide, init_hint

CFG = DecideConfig()


def det(conf: float, cx: float = 0.5, cy: float = 0.5, cls: int = 0) -> Detection:
    return Detection(cls, cx, cy, 0.2, 0.2, conf)


def test_c1_no_detections() -> None:
    result = decide([], CFG)
    assert result.verdict == "review"
    assert result.reason == "kein_objekt"


def test_c2_border_049_is_review_band() -> None:
    result = decide([det(0.49)], CFG)
    assert result.verdict == "review"
    assert result.reason == "grenzfall"


def test_c2_border_050_is_accept() -> None:
    assert decide([det(0.50)], CFG).verdict == "accept"


def test_c3_duplicate_iou_08_review() -> None:
    a = Detection(1, 0.5, 0.5, 0.4, 0.4, 0.9)
    b = Detection(1, 0.51, 0.51, 0.4, 0.4, 0.9)
    result = decide([a, b], CFG)
    assert result.verdict == "review"
    assert result.reason == "doppelt"


def test_c3_iou_06_accept() -> None:
    a = Detection(1, 0.40, 0.5, 0.2, 0.2, 0.9)
    b = Detection(1, 0.55, 0.5, 0.2, 0.2, 0.9)
    assert decide([a, b], CFG).verdict == "accept"


def test_c4_too_many_boxes() -> None:
    dets = [Detection(1, 0.01 + i * 0.001, 0.5, 0.001, 0.001, 0.9) for i in range(51)]
    result = decide(dets, CFG)
    assert result.verdict == "review"
    assert result.reason == "zu_viele"


def test_c5_conf_below_review() -> None:
    result = decide([det(0.20)], CFG)
    assert result.verdict == "review"
    assert result.reason == "unsicher"


def test_c6_accept_never_below_conf_review() -> None:
    for conf in (0.0, 0.1, 0.249, 0.25):
        assert decide([det(conf)], CFG).verdict == "review"


@settings(max_examples=200, deadline=None)
@given(
    confs=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), max_size=6),
    cxs=st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=6, max_size=6),
)
def test_c6_deterministic(confs: list[float], cxs: list[float]) -> None:
    dets = [Detection(1, cxs[i], 0.5, 0.1, 0.1, confs[i]) for i in range(len(confs))]
    assert decide(dets, CFG) == decide(dets, CFG)


def test_thresholds_come_from_config() -> None:
    cfg = DecideConfig(conf_accept=0.9, conf_review=0.1, max_boxes=2, iou_dup=0.5)
    assert decide([det(0.5)], cfg).reason == "grenzfall"
    assert decide([det(0.05)], cfg).reason == "unsicher"
    assert decide([det(0.95), det(0.95, cx=0.05), det(0.95, cx=0.9)], cfg).reason == "zu_viele"


@pytest.mark.parametrize(
    ("n", "init", "must_contain"),
    [
        (8, "scratch", ["8", "pretrained", "meist bessere Ergebnisse"]),
        (8, "pretrained", ["8", "pretrained", "passend bei kleinen Datenmengen"]),
        (250, "scratch", ["250", "scratch", "ausreichend Daten"]),
        (250, "pretrained", ["250", "scratch", "schnellerer Start"]),
    ],
)
def test_e7_init_hint(n: int, init: str, must_contain: list[str]) -> None:
    text = init_hint(n, init)
    for needle in must_contain:
        assert needle in text
