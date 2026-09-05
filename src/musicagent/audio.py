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

    Returns a dict with `bpm` (float), and, when the detected key parses,
    `camelot` (str) and `key_confidence` (float, Essentia's key strength,
    0..1 -- key detection is roughly 70-80% accurate, hence surfacing this).
    Returns `{}` on any failure: no ffmpeg on PATH, essentia not installed,
    a decode error, or an unparseable key. Never raises.
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
        # tmpdir (and both the mp3/wav inside it) is always removed here,
        # success or failure, by the `with` block above.

    out: dict = {"bpm": float(bpm)}
    try:
        out["camelot"] = parse_camelot(f"{key} {scale}")
        out["key_confidence"] = float(strength)
    except ValueError:
        pass  # unparseable key: bpm alone is still useful, camelot stays absent
    return out
