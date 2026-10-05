import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from wfire import data as D            # noqa: E402
from wfire import training             # noqa: E402
from wfire.models import RiskEngine    # noqa: E402
from wfire.webapp import create_app    # noqa: E402


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """A real (quick) training run on the real dataset, shared by all tests."""
    out = tmp_path_factory.mktemp("artifacts")
    training.run(quick=True, out_dir=out, make_figs=False)
    return out


@pytest.fixture
def engine(artifacts):
    return RiskEngine.load(artifacts / "model_bundle.joblib")


@pytest.fixture
def app(artifacts, tmp_path):
    import shutil
    art = tmp_path / "art"
    shutil.copytree(artifacts, art)                       # retraining rewrites files, so each test gets its own copy
    (art / "figures").mkdir(exist_ok=True)
    (art / "figures" / "fig08_roc.png").write_bytes(b"\x89PNG fake")
    return create_app(artifacts_dir=art, db_path=str(tmp_path / "t.db"))


@pytest.fixture
def client(app):
    return app.test_client()


def now_z():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def payload(**over):
    p = {"timestamp": now_z(), "X": 5, "Y": 5, "FFMC": 90.0, "DMC": 100.0, "DC": 600.0, "ISI": 9.0,
         "temp": 25.0, "RH": 35.0, "wind": 4.0, "rain": 0.0}
    p.update(over)
    return p
