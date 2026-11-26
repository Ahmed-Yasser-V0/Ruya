import os
import sys
import shutil
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


@pytest.fixture
def empty_faces(tmp_path):
    """Run face tests against a temporary, empty face database."""
    import face_function as F
    old = F.FACES_DIR, F.DB_PATH
    F.FACES_DIR = str(tmp_path / "faces"); F.DB_PATH = os.path.join(F.FACES_DIR, "face_db.npz")
    yield F
    F.FACES_DIR, F.DB_PATH = old


@pytest.fixture
def client():
    import app as A
    A.KEEP_UPLOADS = False
    return A.app.test_client()
