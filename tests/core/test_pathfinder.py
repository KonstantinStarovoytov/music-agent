import pytest

from musicagent.core.pathfinder import find_path, target_energy
from musicagent.models import Track, TrackRef


def t(title, bpm, camelot, energy):
    return Track(ref=TrackRef(artist="x", title=title), bpm=bpm, camelot=camelot, energy=energy)


def test_target_energy_build_is_monotonic():
    vals = [target_energy("build", i, 10) for i in range(10)]
    assert vals == sorted(vals) and vals[0] < 0.35 and vals[-1] == 1.0


def test_target_energy_unknown_shape_raises():
    with pytest.raises(ValueError):
        target_energy("nonsense", 0, 10)


def test_target_energy_wave_bounded_for_every_position():
    for i in range(10):
        v = target_energy("wave", i, 10)
        assert 0.0 <= v <= 1.0


def test_find_path_orders_by_energy_for_build():
    tracks = [
        t("low", 128, "8A", 0.2),
        t("mid", 128, "8A", 0.5),
        t("high", 129, "9A", 0.9),
    ]
    path = find_path(tracks, "build")
    titles = [tr.ref.title for tr in path.tracks]
    assert titles == ["low", "mid", "high"]
    assert len(path.edge_scores) == 2


def test_isolated_track_excluded():
    tracks = [t("a", 128, "8A", 0.5), t("b", 128, "9A", 0.5), t("island", 175, "3B", 0.5)]
    path = find_path(tracks, "peak_end")
    assert all(tr.ref.title != "island" for tr in path.tracks)


def test_find_path_empty_tracks_returns_empty_path():
    path = find_path([], "build")
    assert path.tracks == []
    assert path.edge_scores == []


def test_find_path_single_track_returns_it_alone():
    tracks = [t("solo", 128, "8A", 0.5)]
    path = find_path(tracks, "build")
    assert len(path.tracks) == 1
    assert path.tracks[0].ref.title == "solo"
    assert path.edge_scores == []


def test_find_path_no_compatible_tracks_returns_single_track_path():
    # Every pair is BPM- or key-incompatible, so no edges exist at all.
    tracks = [
        t("a", 128, "8A", 0.5),
        t("b", 175, "3B", 0.5),
        t("c", 90, "11B", 0.5),
    ]
    path = find_path(tracks, "wave")
    assert len(path.tracks) == 1
    assert path.edge_scores == []
    assert path.tracks[0].ref.title in {"a", "b", "c"}


def test_find_path_never_revisits_a_track():
    tracks = [
        t("a", 128, "8A", 0.2),
        t("b", 128, "8A", 0.5),
        t("c", 128, "8A", 0.8),
        t("d", 128, "8A", 1.0),
    ]
    path = find_path(tracks, "build")
    titles = [tr.ref.title for tr in path.tracks]
    assert len(titles) == len(set(titles))
    assert len(path.edge_scores) == len(path.tracks) - 1


def test_find_path_prefers_longer_path_over_higher_scoring_shorter_path():
    # x/y are identical tracks: their edge score is maxed out (1.0), so a
    # naive "highest total score" search would stop at the 2-track x->y set.
    # m/n/o form a separate, lower-scoring but longer chain in an unrelated
    # key group. The pathfinder must prefer the longer 3-track set.
    x = t("x", 120, "1A", 0.5)
    y = t("y", 120, "1A", 0.5)
    m = t("m", 100, "6A", 0.1)
    n = t("n", 106, "6B", 0.9)
    o = t("o", 112, "7B", 0.1)
    tracks = [x, y, m, n, o]

    path = find_path(tracks, "build")

    assert len(path.tracks) == 3
    assert {tr.ref.title for tr in path.tracks} == {"m", "n", "o"}


def test_find_path_longest_path_survives_beam_truncation():
    # Same x/y vs. m/n/o setup as the test above, plus three extra isolated
    # decoy pairs (each mutually incompatible in key with everything else, so
    # they can never chain past 2 tracks). This brings round 1's candidate
    # count to 12, well past beam_width=4, so the beam_width slice genuinely
    # discards most round-1 candidates -- the m/n/o chain must survive that
    # truncation and go on to win as the longest path.
    x = t("x", 120, "1A", 0.5)
    y = t("y", 120, "1A", 0.5)
    m = t("m", 100, "6A", 0.1)
    n = t("n", 106, "6B", 0.9)
    o = t("o", 112, "7B", 0.1)
    d1a = t("d1a", 100, "9A", 0.1)
    d1b = t("d1b", 105.8, "10A", 0.95)
    d2a = t("d2a", 130, "11A", 0.1)
    d2b = t("d2b", 137.6, "12A", 0.95)
    d3a = t("d3a", 150, "2B", 0.1)
    d3b = t("d3b", 158.9, "3B", 0.95)
    tracks = [x, y, m, n, o, d1a, d1b, d2a, d2b, d3a, d3b]

    path = find_path(tracks, "build", beam_width=4)

    assert len(path.tracks) == 3
    assert {tr.ref.title for tr in path.tracks} == {"m", "n", "o"}
