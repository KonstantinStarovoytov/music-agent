"""Audio analysis of Deezer preview clips: a fallback enrichment provider for
tracks that open metadata APIs (Deezer's own tags, GetSongBPM, MusicBrainz/
AcousticBrainz) don't cover -- mostly small-label/underground releases that
were never catalogued or analysed by those services, but that Deezer still
carries a public 30-second preview clip for.

`essentia` (the actual DSP/ML library doing key + BPM estimation) is an
optional, heavy dependency (~190MB RSS just to import it) and needs the
`ffmpeg` binary on PATH to decode the mp3 preview -- neither is a hard
requirement to run this project, so both are imported/invoked lazily, INSIDE
`analyze_preview`, never at module import time. A deployment that skips the
`audio` extra and/or doesn't have ffmpeg installed still runs fully on the
other (metadata) providers; this module just contributes nothing.
"""

import logging
import os
import shutil
import subprocess
import tempfile

from musicagent.core.camelot import parse_camelot

logger = logging.getLogger(__name__)

# Energy calibration: a heuristic proxy for perceived intensity, not a
# physical quantity. Combines integrated loudness (EBU R128) with rhythmic
# density (onset rate), each normalised to 0..1 and clamped, then blended.
# The floor/ceiling/cap/weights below were picked by measuring both raw
# features (LUFS, onsets/sec) across 8 real tracks spanning house/techno/
# rock/pop and choosing values that gave a sensible spread (0.49-0.74 on that
# set) rather than clustering everything near 0 or 1 -- there is no
# standardized "energy" scale to calibrate against, so this is a judgment
# call, not a derived constant.
LOUDNESS_LUFS_FLOOR = -20.0  # integrated LUFS mapping to loudness_norm 0.0
LOUDNESS_LUFS_CEIL = -5.0  # integrated LUFS mapping to loudness_norm 1.0
ONSET_RATE_CAP = 8.0  # onsets/sec mapping to onset_norm 1.0
ENERGY_WEIGHT_LOUDNESS = 0.6
ENERGY_WEIGHT_ONSET = 0.4


def _energy_from_features(integrated_lufs: float, onset_rate: float) -> float:
    """Pure normalise-and-blend step of the energy heuristic (see the module
    constants above for where the calibration comes from). Kept separate from
    the essentia calls in `_measure_energy` below so it can be unit-tested
    directly against measured (LUFS, onset_rate) pairs without needing
    essentia installed or any real audio."""
    loudness_norm = min(
        max(
            (integrated_lufs - LOUDNESS_LUFS_FLOOR)
            / (LOUDNESS_LUFS_CEIL - LOUDNESS_LUFS_FLOOR),
            0.0,
        ),
        1.0,
    )
    onset_norm = min(max(onset_rate / ONSET_RATE_CAP, 0.0), 1.0)
    energy = ENERGY_WEIGHT_LOUDNESS * loudness_norm + ENERGY_WEIGHT_ONSET * onset_norm
    return min(max(energy, 0.0), 1.0)


def _measure_energy(y, sample_rate: int, es_module) -> float | None:
    """Estimate perceived energy from a decoded mono waveform `y`, combining
    integrated loudness (LoudnessEBUR128, which needs a stereo (N, 2) float32
    array -- built here by duplicating the mono signal) with onset rate
    (OnsetRate, onsets per second). Returns None if either essentia call
    raises (e.g. a clip too short for EBU R128's gating window), so the
    caller can fall back to no energy reading rather than propagating a
    partial/garbage value.
    """
    try:
        # Lazily imported, like essentia itself (see module docstring): numpy
        # is not a declared dependency of this project outside the optional
        # `audio` extra, so it must not be required at module import time.
        import numpy as np

        stereo = np.array([y, y]).T.astype(np.float32)
        integrated_lufs = es_module.LoudnessEBUR128()(stereo)[2]
        onsets, _ = es_module.OnsetRate()(y)
        onset_rate = len(onsets) / (len(y) / sample_rate)
    except Exception:
        logger.warning("essentia energy measurement failed", exc_info=True)
        return None

    return _energy_from_features(integrated_lufs, onset_rate)


# Emitted at most once per process: importing essentia (or missing ffmpeg) is
# an environment-level fact that won't change mid-run, so repeating the
# warning on every track would just be noise for a whole batch/deployment.
_warned_unavailable = False


def _warn_once(message: str) -> None:
    global _warned_unavailable
    if not _warned_unavailable:
        logger.warning(message)
        _warned_unavailable = True


def analyze_preview(mp3_bytes: bytes) -> dict:
    """Decode an mp3 preview clip and estimate its musical key and BPM.

    Blocking/CPU-bound (ffmpeg subprocess + essentia analysis) -- callers on
    an async path must run this via `asyncio.to_thread`, not await it inline.

    Returns a dict with `bpm` (float), `energy` (float, 0..1 -- see
    `_energy_from_features` for the loudness/onset-rate blend that derives
    it; omitted if the essentia measurement itself fails), and, when the
    detected key parses, `camelot` (str) and `key_confidence` (float,
    Essentia's key strength, 0..1 -- key detection is roughly 70-80%
    accurate, hence surfacing this). Returns `{}` on any failure: no ffmpeg
    on PATH, essentia not installed, a decode error, or an unparseable key.
    Never raises.
    """
    if shutil.which("ffmpeg") is None:
        _warn_once("audio analysis provider disabled: ffmpeg binary not found on PATH")
        return {}

    try:
        import essentia
        import essentia.standard as es
    except Exception:  # noqa: BLE001 - essentia's own import can fail in ways
        # other than ModuleNotFoundError (e.g. a missing shared library), and
        # any of them must degrade to {} the same way, never crash the caller.
        _warn_once(
            "audio analysis provider disabled: essentia is not installed "
            "(install the 'audio' extra to enable it)"
        )
        return {}

    essentia.log.infoActive = False
    essentia.log.warningActive = False

    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = os.path.join(tmpdir, "preview.mp3")
        wav_path = os.path.join(tmpdir, "preview.wav")
        with open(mp3_path, "wb") as f:
            f.write(mp3_bytes)

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    mp3_path,
                    "-ac",
                    "1",
                    "-ar",
                    "44100",
                    wav_path,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (subprocess.SubprocessError, OSError):
            logger.warning("ffmpeg failed to decode preview clip", exc_info=True)
            return {}

        try:
            y = es.MonoLoader(filename=wav_path, sampleRate=44100)()
            key, scale, strength = es.KeyExtractor(profileType="edma")(y)
            bpm = es.RhythmExtractor2013(method="multifeature")(y)[0]
        except Exception:
            logger.warning("essentia analysis of preview clip failed", exc_info=True)
            return {}
        energy = _measure_energy(y, 44100, es)
        # tmpdir (and both the mp3/wav inside it) is always removed here,
        # success or failure, by the `with` block above.

    out: dict = {"bpm": float(bpm)}
    if energy is not None:
        out["energy"] = energy
    try:
        out["camelot"] = parse_camelot(f"{key} {scale}")
        out["key_confidence"] = float(strength)
    except ValueError:
        pass  # unparseable key: bpm alone is still useful, camelot stays absent
    return out
