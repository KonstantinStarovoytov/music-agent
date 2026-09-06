import pytest

from musicagent.core.camelot import transition
from musicagent.core.pathfinder import find_path, shift_fit, target_energy
from musicagent.models import Track, TrackRef


def t(title, bpm, camelot, energy):
    return Track(
        ref=TrackRef(artist="x", title=title), bpm=bpm, camelot=camelot, energy=energy
    )


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


def test_shift_fit_follows_the_curve():
    # build rises every step: a boost is aligned, a drop is opposed.
    assert shift_fit(3, "build", 1, 10) == 1.0
    assert shift_fit(-3, "build", 1, 10) == -1.0
    assert shift_fit(1, "build", 1, 10) == pytest.approx(1 / 3)
    # neutral transitions and the first position contribute nothing.
    assert shift_fit(0, "build", 1, 10) == 0.0
    assert shift_fit(3, "build", 0, 10) == 0.0
    # peak_end is flat after the peak: pushing there costs a little.
    assert shift_fit(2, "peak_end", 9, 10) == pytest.approx(-1 / 3)
    # wave falls in its second half: a drop is what the curve wants there.
    assert shift_fit(-2, "wave", 4, 10) > 0.0


def test_build_prefers_a_key_boost_over_a_key_drop_when_energy_ties():
    # Same BPM, same measured energy, so track energy and edge smoothness
    # cannot break the tie; only the harmonic push differs. From 8A, 9A is a
    # boost + and 7A is a drop -: a build should go up.
    tracks = [
        t("start", 128, "8A", 0.5),
        t("up", 128, "9A", 0.5),
        t("down", 128, "7A", 0.5),
    ]
    path = find_path(tracks, "build")
    titles = [tr.ref.title for tr in path.tracks]
    # 7A -> 8A -> 9A: two boost + steps, the only order that climbs twice.
    assert titles == ["down", "start", "up"]
    # And the mirror image: peak_end's flat tail should not reverse it into
    # a chain of drops either -- every edge in the chosen path is a boost.
    labels = [
        transition(a.camelot, b.camelot).label
        for a, b in zip(path.tracks, path.tracks[1:])
    ]
    assert labels == ["energy boost +", "energy boost +"]


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
    tracks = [
        t("a", 128, "8A", 0.5),
        t("b", 128, "9A", 0.5),
        t("island", 175, "3B", 0.5),
    ]
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
    # Decoy BPMs sit well outside every other pair's +-6% window: with the full
    # transition table, keys alone no longer isolate them (9A -> 6A is a
    # boost ++), so tempo has to.
    d1a = t("d1a", 140, "9A", 0.1)
    d1b = t("d1b", 148.4, "10A", 0.95)
    d2a = t("d2a", 160, "11A", 0.1)
    d2b = t("d2b", 169.6, "12A", 0.95)
    d3a = t("d3a", 180, "2B", 0.1)
    d3b = t("d3b", 190.8, "3B", 0.95)
    tracks = [x, y, m, n, o, d1a, d1b, d2a, d2b, d3a, d3b]

    path = find_path(tracks, "build", beam_width=4)

    assert len(path.tracks) == 3
    assert {tr.ref.title for tr in path.tracks} == {"m", "n", "o"}
