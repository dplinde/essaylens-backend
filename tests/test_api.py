import io
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_essaylense.db"
os.environ["ANTHROPIC_API_KEY"] = ""

import pytest
from fastapi.testclient import TestClient

from app.config import settings

settings.anthropic_api_key = None
from app.db import Base, close_db_connections, engine
from app.main import app

ESSAY = (
    "Climate change is one of the defining challenges of our time. According to NASA, global temperatures have risen "
    "by about 1.1 percent since the late nineteenth century, and 97% of climate scientists agree that humans are the cause.\n\n"
    "However, some critics argue that economic costs of action are too high. On the other hand, the cost of inaction is far "
    "greater, for example in flooding, drought, and lost harvests across many regions of the world today.\n\n"
    "Therefore, governments must invest in renewable energy, and individuals should reduce their emissions. "
    "In conclusion, decisive action now will protect future generations from the worst consequences of a warming planet."
)


@pytest.fixture(scope="module")
def client():
    Base.metadata.drop_all(engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(engine)
    close_db_connections()
    if os.path.exists("test_essaylense.db"):
        os.remove("test_essaylense.db")


@pytest.fixture(scope="module")
def auth(client):
    r = client.post("/auth/signup", json={"full_name": "Juan Dela Cruz", "email": "jdcruz@email.com",
                                          "password": "password123", "confirm_password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_signup_validation(client):
    r = client.post("/auth/signup", json={"full_name": "A B", "email": "x@y.com", "password": "password123", "confirm_password": "nope"})
    assert r.status_code == 422


def test_root_redirects_to_api_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_duplicate_and_login(client, auth):
    r = client.post("/auth/signup", json={"full_name": "Juan", "email": "jdcruz@email.com", "password": "password123", "confirm_password": "password123"})
    assert r.status_code == 409
    assert client.post("/auth/login", json={"email": "jdcruz@email.com", "password": "bad"}).status_code == 401
    assert client.post("/auth/login", json={"email": "jdcruz@email.com", "password": "password123"}).status_code == 200


def test_requires_auth(client):
    assert client.get("/essays").status_code == 401


def test_rubric_crud(client, auth):
    rubrics = client.get("/rubrics", headers=auth).json()
    assert rubrics and rubrics[0]["is_builtin"]
    bad = {"rubric_name": "Bad", "criteria": [{"name": "A", "max_score": 10, "weight": 50}]}
    assert client.post("/rubrics", json=bad, headers=auth).status_code == 422
    good = {"rubric_name": "Mine", "scale_type": "likert", "criteria": [
        {"name": "Content", "max_score": 5, "weight": 60}, {"name": "Grammar", "max_score": 5, "weight": 40}]}
    r = client.post("/rubrics", json=good, headers=auth)
    assert r.status_code == 201
    rid = r.json()["rubric_id"]
    good["rubric_name"] = "Mine v2"
    assert client.put(f"/rubrics/{rid}", json=good, headers=auth).json()["version"] == 2
    assert client.put(f"/rubrics/{rubrics[0]['rubric_id']}", json=good, headers=auth).status_code == 404


def test_submit_text_grade_history_revise(client, auth):
    r = client.post("/essays", data={"topic": "Climate change essay", "text": ESSAY}, headers=auth)
    assert r.status_code == 201, r.text
    d = r.json()
    assert 0 <= d["result"]["overall_score"] <= 100
    assert [c["name"] for c in d["result"]["criterion_scores"]] == ["Content", "Organization", "Grammar", "Style"]
    assert d["result"]["suggestions"]

    hist = client.get("/essays", headers=auth).json()
    assert len(hist) == 1 and client.get("/essays", params={"q": "climate"}, headers=auth).json()

    rev = client.post(f"/essays/{d['submission_id']}/revise", json={"text": ESSAY + " Additionally, sources such as (IPCC, 2023) confirm this."}, headers=auth)
    assert rev.status_code == 201 and rev.json()["version"] == 2 and rev.json()["parent_id"] == d["submission_id"]
    cmp = client.get(f"/essays/{d['submission_id']}/compare/{rev.json()['submission_id']}", headers=auth).json()
    assert "overall_delta" in cmp
    assert client.post(f"/essays/{rev.json()['submission_id']}/finalize", headers=auth).json()["is_final"] is True
    assert "GRADE:" in client.get(f"/essays/{d['submission_id']}/export", headers=auth).text


def test_file_upload_txt_and_docx(client, auth):
    r = client.post("/essays", data={"topic": "Climate change"}, files={"file": ("e.txt", ESSAY.encode(), "text/plain")}, headers=auth)
    assert r.status_code == 201 and r.json()["input_method"] == "digital"
    from docx import Document
    buf = io.BytesIO()
    doc = Document()
    for p in ESSAY.split("\n\n"):
        doc.add_paragraph(p)
    doc.save(buf)
    r = client.post("/essays", data={"topic": "Climate change"}, files={"file": ("e.docx", buf.getvalue(), "application/octet-stream")}, headers=auth)
    assert r.status_code == 201


def test_invalid_submissions(client, auth):
    r = client.post("/essays", data={"topic": "Short", "text": "too short"}, headers=auth)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_submission"
    assert client.post("/essays", data={"topic": "x y z"}, files={"file": ("e.xyz", b"hi", "text/plain")}, headers=auth).status_code in (415, 422)
    # blank image => Invalid Submission
    import cv2, numpy as np
    ok, png = cv2.imencode(".png", np.full((600, 800, 3), 255, np.uint8))
    r = client.post("/essays", data={"topic": "Handwritten test"}, files={"file": ("p.png", png.tobytes(), "image/png")}, headers=auth)
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_submission"


def test_isolation_and_analytics(client, auth):
    other = client.post("/auth/signup", json={"full_name": "Other User", "email": "o@email.com", "password": "password123", "confirm_password": "password123"}).json()
    oh = {"Authorization": f"Bearer {other['access_token']}"}
    sid = client.get("/essays", headers=auth).json()[0]["submission_id"]
    assert client.get(f"/essays/{sid}", headers=oh).status_code == 404
    s = client.get("/analytics/summary", headers=auth).json()
    assert s["essay_count"] >= 2 and s["average_score"] and "Content" in s["criterion_averages"]


def test_profile_update_and_password_change(client, auth):
    assert client.put("/auth/me", json={"full_name": "Juan D. Cruz"}, headers=auth).json()["full_name"] == "Juan D. Cruz"
    bad = client.post("/auth/change-password", json={"current_password": "nope", "new_password": "newpassword1"}, headers=auth)
    assert bad.status_code == 400
    ok = client.post("/auth/change-password", json={"current_password": "password123", "new_password": "newpassword1"}, headers=auth)
    assert ok.status_code == 204
    assert client.post("/auth/login", json={"email": "jdcruz@email.com", "password": "newpassword1"}).status_code == 200
    # restore so later tests that log in with the original password stay valid
    client.post("/auth/change-password", json={"current_password": "newpassword1", "new_password": "password123"}, headers=auth)


def test_concise_style_accepted(client, auth):
    r = client.post("/essays", data={"topic": "Climate change", "text": ESSAY, "style": "Concise"}, headers=auth)
    assert r.status_code == 201


def test_delete_account(client):
    t = client.post("/auth/signup", json={"full_name": "Temp User", "email": "t@email.com", "password": "password123", "confirm_password": "password123"}).json()
    h = {"Authorization": f"Bearer {t['access_token']}"}
    client.post("/essays", data={"topic": "Climate change", "text": ESSAY}, headers=h)
    assert client.delete("/auth/me", headers=h).status_code == 204
    assert client.get("/auth/me", headers=h).status_code == 401
