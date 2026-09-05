from musicagent.llm import _Explanations, explain_set, parse_input

from musicagent.models import SetPath, SetRequest, Track, TrackRef


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self.result


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
    """Test that explain_set on a path with 1 track returns no transitions."""
    tracks = [Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A")]
    path = SetPath(tracks=tracks, edge_scores=[])
    fake = FakeLLM(_Explanations(explanations=[], summary="single track"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 0
    assert result.summary == "single track"


def test_explain_set_empty_path():
    """Test that explain_set on an empty path returns no transitions."""
    path = SetPath(tracks=[], edge_scores=[])
    fake = FakeLLM(_Explanations(explanations=[], summary="empty"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 0
    assert result.summary == "empty"


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


def test_parse_input_with_default_llm_not_called_at_import():
    """Test that importing doesn't call get_llm() or fail with no API key."""
    # This test just verifies the import succeeds
    from musicagent import llm as llm_module

    assert hasattr(llm_module, "get_llm")
    assert hasattr(llm_module, "parse_input")
    assert hasattr(llm_module, "explain_set")
