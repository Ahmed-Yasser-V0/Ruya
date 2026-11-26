"""API tests through Flask's test client (every mode, both languages)."""
import pytest


def post(client, mode, image="images/car_person.jpg", **form):
    data = {"select_mode": mode, **form}
    if image:
        data["image"] = (open(image, "rb"), "frame.jpg")
    r = client.post("/detect", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    return r.get_json()[0]


@pytest.mark.parametrize("lang", ["en", "ar"])
@pytest.mark.parametrize("mode", ["object", "currency", "face", "list_faces", "text", "describe", "summarize", "bogus"])
def test_every_mode_answers(client, empty_faces, mode, lang):
    out = post(client, mode, lang=lang)
    assert out["result"] and out["mode"] == mode and out["lang"] == lang


def test_detect_returns_boxes(client):
    out = post(client, "object")
    assert out["detections"] and all(0 <= v <= 1 for d in out["detections"] for v in d["box"])


def test_find_with_natural_words(client):
    assert "o'clock" in post(client, "find", object_to_be_found="The Car")["result"]


def test_no_image(client):
    r = client.post("/detect", data={"select_mode": "object"})
    assert "No image" in r.get_json()[0]["result"]


def test_guide_endpoint(client):
    r = client.post("/guide", data={"session": "t", "image": (open("images/car_person.jpg", "rb"), "g.jpg")},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert "events" in body and body["detections"]
    assert client.post("/guide/reset", data={"session": "t"}).get_json()["ok"]


def test_health(client):
    assert client.get("/health").get_json()["ok"] is True


def test_debugger_is_off():
    import app as A
    assert not A.app.debug
