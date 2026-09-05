import os

from pydantic import BaseModel

from musicagent.models import (
    SetPath,
    SetRequest,
    SetResult,
    TrackRef,
    Transition,
    UnresolvedTrack,
)


def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("MUSICAGENT_MODEL", "gpt-4o-mini"), temperature=0
    )


class _Explanations(BaseModel):
    explanations: list[str]
    summary: str


class LLMOutputError(RuntimeError):
    """Raised when an LLM node fails to produce valid structured output,
    even after one retry (spec.md section 6)."""


PARSE_PROMPT = """You parse DJ set requests. Extract the track list (artist + title),
the desired energy shape (build / peak_end / wave; default peak_end), and, if the
request names a target set length (e.g. "a 45 minute set", "about an hour"), the
duration in minutes as duration_min.
User request:
{text}"""

EXPLAIN_PROMPT = """You are a DJ explaining a set. For each consecutive pair of tracks
below, write one short explanation of why the transition works (key relationship on the
Camelot wheel, BPM closeness, energy movement). Then a 1-2 sentence summary of the set arc.
Some tracks show a key confidence (0-1, from algorithmic key detection); when it's low
(below ~0.5), hedge the key claim in your explanation instead of stating it flatly.
If a list of unresolved or left-out tracks appears below, briefly mention those tracks in
the summary in plain language, using only the reasons given there -- never invent a
different reason. If no such list appears, every track the user gave made it into the set:
say nothing about tracks being skipped, dropped or left out, and do not speculate that any
were. Claiming omissions that did not happen is worse than saying nothing.
Return exactly {n} explanations, in order.

Tracks (in play order, with camelot/bpm/energy):
{tracks}{omitted_note}{unresolved_note}"""


def _invoke_structured(llm, schema: type, prompt: str, node: str):
    """Invoke `llm.with_structured_output(schema).invoke(prompt)`, validating the
    result is an instance of `schema`. Retries exactly once on any failure
    (exception, None, wrong type) by replaying the identical prompt -- this is
    a plain retry, not a repair: there is no error feedback given back to the
    model, since the model never sees why the first attempt was rejected.
    Raises LLMOutputError if the retry also fails.
    """

    def _attempt():
        try:
            result = llm.with_structured_output(schema).invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see spec 6
            return None, exc
        if not isinstance(result, schema):
            return None, TypeError(
                f"expected {schema.__name__}, got {type(result).__name__}"
            )
        return result, None

    result, err = _attempt()
    if result is not None:
        return result

    result, err2 = _attempt()
    if result is not None:
        return result

    raise LLMOutputError(
        f"{node}: LLM failed to produce valid {schema.__name__} output after "
        f"one retry (last error: {err2 or err})"
    )


def parse_input(text: str, llm=None) -> SetRequest:
    llm = llm or get_llm()
    request = _invoke_structured(
        llm, SetRequest, PARSE_PROMPT.format(text=text), node="parse_input"
    )
    if not request.tracks:
        raise LLMOutputError("parse_input: no tracks could be parsed from the request")
    return request


def explain_set(
    path: SetPath,
    unresolved: list[UnresolvedTrack],
    omitted: list[TrackRef] | None = None,
    llm=None,
) -> SetResult:
    omitted = omitted or []
    if len(path.tracks) < 2:
        return SetResult(
            transitions=[],
            summary="No transitions to explain.",
            unresolved=unresolved,
            omitted=omitted,
        )

    llm = llm or get_llm()
    pairs = list(zip(path.tracks, path.tracks[1:]))

    def _track_line(i: int, t) -> str:
        conf = (
            f", key confidence {t.key_confidence:.2f}"
            if t.key_confidence is not None
            else ""
        )
        return (
            f"{i + 1}. {t.ref.artist} - {t.ref.title} "
            f"[{t.camelot}, {t.bpm:.0f} BPM, energy {t.energy:.2f}{conf}]"
        )

    lines = "\n".join(_track_line(i, t) for i, t in enumerate(path.tracks))
    # Only add this line (and its tokens) when there's something to say --
    # most requests place every resolved track, so this costs nothing then.
    omitted_note = ""
    if omitted:
        names = ", ".join(f"{r.artist} - {r.title}" for r in omitted)
        omitted_note = f"\n\n(Left out of the set, though resolved fine: {names}.)"
    # Only add this line (and its tokens) when there's something to say --
    # most requests resolve every track, so this costs nothing then. The
    # reason text comes from UnresolvedTrack.message (deterministic, code-set)
    # so the LLM only ever paraphrases it, never invents its own reason.
    unresolved_note = ""
    if unresolved:
        names = ", ".join(f"{u.artist} - {u.title} ({u.message})" for u in unresolved)
        unresolved_note = (
            f"\n\n(Could not be resolved and are not in the set: {names}.)"
        )
    out = _invoke_structured(
        llm,
        _Explanations,
        EXPLAIN_PROMPT.format(
            n=len(pairs),
            tracks=lines,
            omitted_note=omitted_note,
            unresolved_note=unresolved_note,
        ),
        node="explain_set",
    )
    raw_explanations = out.explanations or []
    explanations = [str(e) for e in raw_explanations]
    exps = (explanations + [""] * len(pairs))[: len(pairs)]
    transitions = [
        Transition(from_track=a.ref, to_track=b.ref, explanation=e)
        for (a, b), e in zip(pairs, exps)
    ]
    return SetResult(
        transitions=transitions,
        summary=out.summary,
        unresolved=unresolved,
        omitted=omitted,
    )
