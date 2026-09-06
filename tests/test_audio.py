import sys
import types

import pytest

from musicagent.audio import (
    AMBIGUOUS_MODE_PENALTY,
    _energy_from_features,
    analyze_preview,
    vote_key,
)

# Measured (LUFS, onsets/sec) -> expected energy pairs, from real tracks
# (see musicagent.audio module docstring / bug report for calibration
# provenance): 0.6 * loudness_norm + 0.4 * onset_norm, clamped 0..1.
MEASURED_TRACKS = [
    ("Matt Milano - Black Sea", -8.5, 3.70, 0.65),
    ("A-Brothers - Diabolus", -13.5, 5.77, 0.55),
    ("Electrorites & Dolby D - Project 3", -11.0, 7.40, 0.73),
    ("Bettosun - Antares", -9.9, 6.80, 0.74),
    ("Cortechs - Hollensturz", -9.3, 3.83, 0.62),
    ("Krizz Karo - Bad Reflection", -12.8, 4.07, 0.49),
    ("Radiohead - Creep", -11.3, 3.27, 0.51),
    ("Peggy Gou - It Makes You Forget", -11.7, 5.24, 0.59),
]


@pytest.mark.parametrize("name,lufs,onset_rate,expected", MEASURED_TRACKS)
def test_energy_from_features_matches_measured_tracks(name, lufs, onset_rate, expected):
    assert round(_energy_from_features(lufs, onset_rate), 2) == expected


@pytest.mark.parametrize(
    "lufs,onset_rate",
    [
        (-40.0, 0.0),  # far below floor, no onsets
        (5.0, 0.0),
        (-40.0, 50.0),  # far above cap
        (5.0, 50.0),  # far above both ceiling and cap
    ],
)
def test_energy_from_features_clamps_extreme_inputs(lufs, onset_rate):
    energy = _energy_from_features(lufs, onset_rate)
    assert 0.0 <= energy <= 1.0


def _install_fake_essentia(monkeypatch, lufs: float, onset_rate: float):
    """Build a minimal fake `essentia`/`essentia.standard` pair and install it
    into sys.modules so `analyze_preview`'s lazy `import essentia` /
    `import essentia.standard as es` picks it up without essentia (a heavy
    optional dependency) actually being installed. `MonoLoader`/
    `KeyExtractor`/`RhythmExtractor2013` are stubbed with fixed, valid
    output; `LoudnessEBUR128`/`OnsetRate` are stubbed to report exactly the
    given (lufs, onset_rate) pair, engineered so the real
    `len(onsets) / (len(y) / 44100.0)` computation in `_measure_energy`
    reproduces `onset_rate` to 2 decimals.
    """
    n_onsets = round(onset_rate * 100)
    fake_y = list(range(100 * 44100))  # 100s of "audio" -> onset_rate = n_onsets / 100

    class _MonoLoader:
        def __init__(self, **kwargs):
            pass

        def __call__(self):
            return fake_y

    class _KeyExtractor:
        def __init__(self, **kwargs):
            pass

        def __call__(self, y):
            return ("C", "major", 0.8)

    class _RhythmExtractor2013:
        def __init__(self, **kwargs):
            pass

        def __call__(self, y):
            return (128.0, None, None, None, None)

    class _LoudnessEBUR128:
        def __call__(self, stereo):
            return (None, None, lufs)

    class _OnsetRate:
        def __call__(self, y):
            return (list(range(n_onsets)), None)

    fake_es_standard = types.SimpleNamespace(
        MonoLoader=_MonoLoader,
        KeyExtractor=_KeyExtractor,
        RhythmExtractor2013=_RhythmExtractor2013,
        LoudnessEBUR128=_LoudnessEBUR128,
        OnsetRate=_OnsetRate,
    )
    fake_essentia = types.SimpleNamespace(
        log=types.SimpleNamespace(infoActive=True, warningActive=True),
        standard=fake_es_standard,
    )
    monkeypatch.setitem(sys.modules, "essentia", fake_essentia)
    monkeypatch.setitem(sys.modules, "essentia.standard", fake_es_standard)


@pytest.mark.parametrize("name,lufs,onset_rate,expected", MEASURED_TRACKS)
def test_analyze_preview_energy_matches_measured_tracks(
    monkeypatch, name, lufs, onset_rate, expected
):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
    _install_fake_essentia(monkeypatch, lufs, onset_rate)

    result = analyze_preview(b"fake mp3 bytes")

    assert 0.0 <= result["energy"] <= 1.0
    assert round(result["energy"], 2) == expected


# --- key mode voting (spec section 3, "Key mode voting") ---------------------


def test_vote_key_unanimous_keeps_mode_and_mean_strength():
    camelot, conf, ambiguous = vote_key(
        [("6B", 0.86), ("6B", 0.89), ("6B", 0.90), ("6B", 0.92)]
    )
    assert (camelot, ambiguous) == ("6B", False)
    assert conf == pytest.approx((0.86 + 0.89 + 0.90 + 0.92) / 4)


def test_vote_key_mode_split_prefers_minor_with_penalty():
    # Kolsch - Grey as measured: edma says 8A, three general profiles say 8B.
    camelot, conf, ambiguous = vote_key(
        [("8A", 0.78), ("8B", 0.82), ("8B", 0.81), ("8B", 0.83)]
    )
    assert (camelot, ambiguous) == ("8A", True)
    mean = (0.78 + 0.82 + 0.81 + 0.83) / 4
    assert conf == pytest.approx(mean * AMBIGUOUS_MODE_PENALTY)


def test_vote_key_number_split_scales_confidence_by_share():
    # Adriatique - Deep In The Three as measured: numbers all over the place.
    camelot, conf, ambiguous = vote_key(
        [("8B", 0.77), ("6A", 0.77), ("5A", 0.74), ("8B", 0.69)]
    )
    assert camelot == "8B" and ambiguous is False
    assert conf == pytest.approx(((0.77 + 0.69) / 2) * 0.5)


def test_vote_key_number_tie_breaks_on_strength():
    camelot, _, _ = vote_key([("8B", 0.9), ("6A", 0.5), ("6A", 0.5), ("8B", 0.9)])
    assert camelot == "8B"


def test_vote_key_no_votes():
    assert vote_key([]) is None


def test_analyze_preview_votes_across_profiles(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
    _install_fake_essentia(monkeypatch, lufs=-10.0, onset_rate=3.0)

    readings = {
        "edma": ("A", "minor", 0.78),
        "bgate": ("C", "major", 0.82),
        "braw": ("C", "major", 0.81),
        "krumhansl": ("C", "major", 0.83),
    }

    class _KeyExtractor:
        def __init__(self, profileType):
            self.profile = profileType

        def __call__(self, y):
            return readings[self.profile]

    sys.modules["essentia.standard"].KeyExtractor = _KeyExtractor

    result = analyze_preview(b"fake mp3 bytes")

    assert result["camelot"] == "8A"
    assert result["key_confidence"] < 0.78
