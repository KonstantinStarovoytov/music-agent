import sys
import types

import pytest

from musicagent.audio import _energy_from_features, analyze_preview

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
