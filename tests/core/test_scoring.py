from musicagent.core.scoring import bpm_ok, build_edges, edge_score
from musicagent.models import Track, TrackRef


def t(title, bpm, camelot, energy=0.5):
    return Track(
        ref=TrackRef(artist="x", title=title), bpm=bpm, camelot=camelot, energy=energy
    )


def test_bpm_window():
    assert bpm_ok(128, 130)  # ~1.6%
    assert not bpm_ok(128, 140)  # ~9%


def test_bpm_window_boundary():
    assert bpm_ok(100, 106)  # exactly 6/106 = 5.66% <= 6%
    assert not bpm_ok(100, 106.4)  # 6.4/106.4 = 6.01% > 6%


def test_incompatible_key_scores_zero():
    assert edge_score(t("a", 128, "8A"), t("b", 128, "3B")) == 0.0


def test_same_key_beats_neighbor():
    base = t("a", 128, "8A")
    assert edge_score(base, t("b", 128, "8A")) > edge_score(base, t("c", 128, "9A"))


def test_build_edges_directed_and_filtered():
    tracks = [t("a", 128, "8A"), t("b", 129, "9A"), t("c", 90, "8A")]
    edges = build_edges(tracks)
    pairs = {(e.a, e.b) for e in edges}
    assert (0, 1) in pairs and (1, 0) in pairs
    assert not any(0 in p and 2 in p for p in pairs)  # bpm too far
