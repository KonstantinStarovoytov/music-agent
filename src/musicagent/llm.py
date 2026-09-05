import os

from pydantic import BaseModel

from musicagent.models import SetPath, SetRequest, SetResult, TrackRef, Transition


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
    even after one repair retry (spec.md section 6)."""


PARSE_PROMPT = """You parse DJ set requests. Extract the track list (artist + title)
and the desired energy shape (build / peak_end / wave; default peak_end).
User request:
{text}"""

EXPLAIN_PROMPT = """You are a DJ explaining a set. For each consecutive pair of tracks
below, write one short explanation of why the transition works (key relationship on the
Camelot wheel, BPM closeness, energy movement). Then a 1-2 sentence summary of the set arc.
Return exactly {n} explanations, in order.

Tracks (in play order, with camelot/bpm/energy):
{tracks}"""


def _invoke_structured(llm, schema: type, prompt: str, node: str):
    """Invoke `llm.with_structured_output(schema).invoke(prompt)`, validating the
    result is an instance of `schema`. Retries exactly once on any failure
    (exception, None, wrong type). Raises LLMOutputError if the retry also fails.
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
        f"one repair retry (last error: {err2 or err})"
    )


def parse_input(text: str, llm=None) -> SetRequest:
    llm = llm or get_llm()
    request = _invoke_structured(
        llm, SetRequest, PARSE_PROMPT.format(text=text), node="parse_input"
    )
    if not request.tracks:
        raise LLMOutputError("parse_input: no tracks could be parsed from the request")
    return request


def explain_set(path: SetPath, unresolved: list[TrackRef], llm=None) -> SetResult:
    if len(path.tracks) < 2:
        return SetResult(
            transitions=[], summary="No transitions to explain.", unresolved=unresolved
        )

    llm = llm or get_llm()
    pairs = list(zip(path.tracks, path.tracks[1:]))
    lines = "\n".join(
        f"{i + 1}. {t.ref.artist} - {t.ref.title} [{t.camelot}, {t.bpm:.0f} BPM, energy {t.energy:.2f}]"
        for i, t in enumerate(path.tracks)
    )
    out = _invoke_structured(
        llm,
        _Explanations,
        EXPLAIN_PROMPT.format(n=len(pairs), tracks=lines),
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
        transitions=transitions, summary=out.summary, unresolved=unresolved
    )
