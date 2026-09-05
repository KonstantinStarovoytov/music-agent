from musicagent.models import SetRequest, TrackRef


def test_set_request_defaults():
    req = SetRequest(tracks=[TrackRef(artist="Bicep", title="Glue")])
    assert req.energy_shape == "peak_end"
    assert req.duration_min is None


def test_energy_shape_validated():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SetRequest(tracks=[], energy_shape="chaotic")
