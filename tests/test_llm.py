import pytest

from musicagent.llm import LLMOutputError, _Explanations, explain_set, parse_input
from musicagent.models import SetPath, SetRequest, Track, TrackRef


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.calls += 1
        return self.result


class RecordingLLM:
    """Captures the rendered prompt passed to invoke()."""

    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.result


class SequenceLLM:
    """Returns/raises a different outcome on each successive invoke() call.

    Each item in `outcomes` is either a value to return, or an Exception
    instance/class to raise.
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception) or (
            isinstance(outcome, type) and issubclass(outcome, Exception)
        ):
            raise outcome
        return outcome


def test_parse_input_returns_request():
    fake = FakeLLM(
        SetRequest(
            tracks=[TrackRef(artist="Bicep", title="Glue")], energy_shape="build"
        )
    )
    req = parse_input("bicep glue, хочу нарастающий сет", llm=fake)
    assert req.energy_shape == "build" and req.tracks[0].title == "Glue"


def test_explain_set_zips_transitions():
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    fake = FakeLLM(_Explanations(explanations=["smooth +1 move"], summary="nice set"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 1
    assert result.transitions[0].to_track.title == "t2"
    assert result.summary == "nice set"


def test_explain_set_pads_explanations():
    """Test that if LLM returns too few explanations, they get padded."""
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
        Track(ref=TrackRef(artist="c", title="t3"), bpm=128, camelot="10A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9, 0.8])
    # Only 1 explanation, but we need 2
    fake = FakeLLM(_Explanations(explanations=["smooth +1 move"], summary="nice set"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 2
    assert result.transitions[0].explanation == "smooth +1 move"
    assert result.transitions[1].explanation == ""


def test_explain_set_truncates_explanations():
    """Test that if LLM returns too many explanations, they get truncated."""
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    # 3 explanations, but we only need 1
    fake = FakeLLM(
        _Explanations(
            explanations=["smooth +1 move", "extra 1", "extra 2"], summary="nice set"
        )
    )
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 1
    assert result.transitions[0].explanation == "smooth +1 move"


def test_explain_set_single_track():
    """explain_set on a path with 1 track returns no transitions and never
    invokes the LLM (there is nothing to explain)."""
    tracks = [Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A")]
    path = SetPath(tracks=tracks, edge_scores=[])
    fake = FakeLLM(_Explanations(explanations=[], summary="single track"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 0
    assert fake.calls == 0


def test_explain_set_empty_path():
    """explain_set on an empty path returns no transitions and never invokes
    the LLM (there is nothing to explain)."""
    path = SetPath(tracks=[], edge_scores=[])
    fake = FakeLLM(_Explanations(explanations=[], summary="empty"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 0
    assert fake.calls == 0


def test_explain_set_preserves_unresolved():
    """Test that unresolved passed in is carried through unchanged."""
    unresolved_refs = [TrackRef(artist="x", title="y"), TrackRef(artist="p", title="q")]
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    fake = FakeLLM(_Explanations(explanations=["smooth"], summary="nice set"))
    result = explain_set(path, unresolved=unresolved_refs, llm=fake)
    assert result.unresolved == unresolved_refs


def test_explain_set_preserves_omitted_and_defaults_to_empty():
    """omitted is carried through when given, and defaults to [] when the
    caller (e.g. run_set with nothing left out) doesn't pass it."""
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    omitted_refs = [TrackRef(artist="c", title="island")]
    fake = FakeLLM(_Explanations(explanations=["smooth"], summary="nice set"))

    result = explain_set(path, unresolved=[], omitted=omitted_refs, llm=fake)
    assert result.omitted == omitted_refs

    result_default = explain_set(path, unresolved=[], llm=fake)
    assert result_default.omitted == []


def test_explain_set_mentions_omitted_tracks_in_prompt_only_when_present():
    """The omitted-tracks note is only added to the prompt when there's
    something to say -- an empty omitted list must cost nothing."""
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])

    recorder = RecordingLLM(_Explanations(explanations=["smooth"], summary="nice"))
    explain_set(path, unresolved=[], llm=recorder)
    assert "island" not in recorder.prompts[0]

    recorder2 = RecordingLLM(_Explanations(explanations=["smooth"], summary="nice"))
    explain_set(
        path,
        unresolved=[],
        omitted=[TrackRef(artist="c", title="island")],
        llm=recorder2,
    )
    assert "island" in recorder2.prompts[0]


def test_parse_input_lazy_llm_no_network_and_no_import_side_effect(monkeypatch):
    """get_llm() must not be invoked at import time, and importing the module
    (and referencing get_llm) must not raise even with no API key set. get_llm
    itself must only be reached when a function is called without an explicit
    llm= (never exercised here, to avoid any real network call)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from musicagent import llm as llm_module

    # Referencing get_llm (not calling it) must not raise.
    assert callable(llm_module.get_llm)

    # Calling parse_input/explain_set with an explicit llm= must never reach
    # get_llm() (and therefore never touch the network / API key).
    def _boom():
        raise AssertionError("get_llm() must not be called when llm= is provided")

    monkeypatch.setattr(llm_module, "get_llm", _boom)

    fake = FakeLLM(
        SetRequest(tracks=[TrackRef(artist="a", title="b")], energy_shape="build")
    )
    result = llm_module.parse_input("x", llm=fake)
    assert result.tracks[0].artist == "a"


# --- Finding 1: repair-retry behavior ---


def test_parse_input_retries_once_then_succeeds():
    good = SetRequest(tracks=[TrackRef(artist="a", title="b")], energy_shape="build")
    seq = SequenceLLM([RuntimeError("boom"), good])
    result = parse_input("some text", llm=seq)
    assert result is good
    assert seq.calls == 2


def test_parse_input_fails_both_times_raises_llm_output_error():
    seq = SequenceLLM([RuntimeError("boom1"), RuntimeError("boom2")])
    with pytest.raises(LLMOutputError):
        parse_input("some text", llm=seq)
    assert seq.calls == 2


def test_parse_input_returns_none_raises_after_retry():
    seq = SequenceLLM([None, None])
    with pytest.raises(LLMOutputError):
        parse_input("some text", llm=seq)
    assert seq.calls == 2


def test_parse_input_wrong_type_raises_after_retry():
    seq = SequenceLLM(["not a SetRequest", "still wrong"])
    with pytest.raises(LLMOutputError):
        parse_input("some text", llm=seq)
    assert seq.calls == 2


def test_parse_input_empty_tracks_raises_llm_output_error():
    empty = SetRequest(tracks=[], energy_shape="build")
    fake = FakeLLM(empty)
    with pytest.raises(LLMOutputError):
        parse_input("gibberish", llm=fake)


def test_explain_set_retries_once_then_succeeds():
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    good = _Explanations(explanations=["ok"], summary="nice")
    seq = SequenceLLM([RuntimeError("boom"), good])
    result = explain_set(path, unresolved=[], llm=seq)
    assert result.summary == "nice"
    assert seq.calls == 2


def test_explain_set_explanations_none_is_coerced():
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    out = _Explanations.model_construct(explanations=None, summary="nice")
    fake = FakeLLM(out)
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 1
    assert result.transitions[0].explanation == ""


def test_explain_set_explanations_non_string_items_coerced():
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    out = _Explanations.model_construct(explanations=[123], summary="nice")
    fake = FakeLLM(out)
    result = explain_set(path, unresolved=[], llm=fake)
    assert result.transitions[0].explanation == "123"


# --- prompt construction coverage ---


def test_explain_set_prompt_contains_track_fields():
    tracks = [
        Track(ref=TrackRef(artist="Bicep", title="Glue"), bpm=128.0, camelot="8A"),
        Track(ref=TrackRef(artist="Four Tet", title="Baby"), bpm=126.0, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    recorder = RecordingLLM(_Explanations(explanations=["smooth"], summary="nice"))
    explain_set(path, unresolved=[], llm=recorder)

    assert len(recorder.prompts) == 1
    prompt = recorder.prompts[0]
    for artist, title, camelot, bpm in [
        ("Bicep", "Glue", "8A", "128"),
        ("Four Tet", "Baby", "9A", "126"),
    ]:
        assert artist in prompt
        assert title in prompt
        assert camelot in prompt
        assert bpm in prompt
