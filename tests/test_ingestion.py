from datetime import datetime, timedelta, timezone

import pytest

from conftest import now_z, payload
from wfire.ingestion import Ingestor, ValidationError

TR = {"temp": [2.2, 33.3], "RH": [15, 100], "wind": [0.4, 9.4], "rain": [0, 6.4], "FFMC": [18.7, 96.2], "DMC": [1.1, 291.3], "DC": [7.9, 860.6], "ISI": [0, 56.1]}
NOW = datetime(2026, 8, 14, 13, 0, tzinfo=timezone.utc)


def fields(exc):
    return {e["field"] for e in exc.value.errors}


def test_valid_reading_is_accepted_and_features_are_in_model_order():
    r = Ingestor(TR).process(payload(timestamp="2026-08-14T13:00:00Z"), now=NOW)
    assert r.zone_id == "X5-Y5" and list(r.features) == ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
    assert r.features["month"] == 8 and r.features["day"] == 4      # Friday, derived from the timestamp
    assert r.warnings == [] and r.filtered == []


def test_every_problem_is_reported_at_once():
    with pytest.raises(ValidationError) as e:
        Ingestor(TR).process({"timestamp": "nonsense", "X": 5}, now=NOW)
    assert {"timestamp", "Y", "FFMC", "temp", "RH"} <= {x["field"] for x in e.value.errors}


@pytest.mark.parametrize("bad", [{"temp": 95}, {"RH": 120}, {"RH": -1}, {"wind": -3}, {"X": 10}, {"Y": 1}, {"X": 5.5},
                                 {"temp": True}, {"temp": float("nan")}, {"temp": float("inf")}, {"temp": "hot"}, {"temp": ""}, {"temp": None}])
def test_impossible_or_malformed_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        Ingestor(TR).process(payload(**bad), allow_stale=True)


def test_numeric_strings_are_accepted():
    assert Ingestor(TR).process(payload(temp="21.5"), allow_stale=True).features["temp"] == 21.5


def test_timestamp_rules():
    ing = Ingestor(TR, max_age_s=3600, max_skew_s=300)
    with pytest.raises(ValidationError) as e:
        ing.process({k: v for k, v in payload().items() if k != "timestamp"}, now=NOW)
    assert fields(e) == {"timestamp"}
    for bad in ("2026-08-14T13:00:00", "14/08/2026", "2026-08-14T15:00:00Z", "2026-08-14T11:00:00Z"):    # naive, garbage, future, stale
        with pytest.raises(ValidationError) as e:
            ing.process(payload(timestamp=bad), now=NOW)
        assert fields(e) == {"timestamp"}, bad
    assert ing.process(payload(timestamp="2026-08-14T11:00:00Z"), now=NOW, allow_stale=True)


def test_timestamps_are_normalised_to_utc():
    r = Ingestor(TR).process(payload(timestamp="2026-08-14T18:30:00+05:30"), now=NOW)
    assert r.timestamp == datetime(2026, 8, 14, 13, 0, tzinfo=timezone.utc)


def test_month_and_day_can_be_names_or_numbers():
    f = Ingestor(TR).validate(payload(month="Sep", day="sun"), allow_stale=True)[0]
    assert (f["month"], f["day"]) == (9, 6)
    with pytest.raises(ValidationError):
        Ingestor(TR).validate(payload(month="smarch"), allow_stale=True)
    with pytest.raises(ValidationError):
        Ingestor(TR).validate(payload(month=13), allow_stale=True)


def test_out_of_training_range_is_a_warning_not_a_rejection():
    r = Ingestor(TR).process(payload(temp=45), allow_stale=True)
    assert any("temp" in w and "extrapolating" in w for w in r.warnings)


def test_spike_is_replaced_by_the_rolling_median_and_does_not_poison_the_window():
    ing = Ingestor(TR)
    for t in (24.0, 24.5, 25.0, 24.2, 24.8):
        assert ing.process(payload(temp=t), allow_stale=True).filtered == []
    spike = ing.process(payload(temp=52.0), allow_stale=True)
    assert spike.filtered and spike.filtered[0]["field"] == "temp" and 24 <= spike.features["temp"] <= 25
    assert any("spike" in w for w in spike.warnings)
    after = ing.process(payload(temp=25.3), allow_stale=True)
    assert after.filtered == [] and after.features["temp"] == 25.3


def test_no_filtering_until_there_is_history_and_zones_are_independent():
    ing = Ingestor(TR)
    assert ing.process(payload(temp=10), allow_stale=True).features["temp"] == 10
    assert ing.process(payload(temp=30), allow_stale=True).filtered == []     # only 1 prior reading: cannot judge yet
    for t in (24, 24.4, 24.2, 24.6, 24.1):
        ing.process(payload(temp=t, X=2, Y=2), allow_stale=True)
    other = ing.process(payload(temp=40, X=7, Y=7), allow_stale=True)        # zone with no history is untouched
    assert other.filtered == []
