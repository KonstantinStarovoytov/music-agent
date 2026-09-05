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


def parse_input(text: str, llm=None) -> SetRequest:
    llm = llm or get_llm()
    return llm.with_structured_output(SetRequest).invoke(PARSE_PROMPT.format(text=text))


def explain_set(path: SetPath, unresolved: list[TrackRef], llm=None) -> SetResult:
    llm = llm or get_llm()
    pairs = list(zip(path.tracks, path.tracks[1:]))
    lines = "\n".join(
        f"{i + 1}. {t.ref.artist} - {t.ref.title} [{t.camelot}, {t.bpm:.0f} BPM, energy {t.energy:.2f}]"
        for i, t in enumerate(path.tracks)
    )
    out = llm.with_structured_output(_Explanations).invoke(
        EXPLAIN_PROMPT.format(n=len(pairs), tracks=lines)
    )
    exps = (out.explanations + [""] * len(pairs))[: len(pairs)]
    transitions = [
        Transition(from_track=a.ref, to_track=b.ref, explanation=e)
        for (a, b), e in zip(pairs, exps)
    ]
    return SetResult(
        transitions=transitions, summary=out.summary, unresolved=unresolved
    )
