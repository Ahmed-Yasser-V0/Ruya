"""Mood, age group and money on CCTV."""
import cv2
import numpy as np

import face_function as F
from core import face_attributes as FA


def test_models_present():
    assert FA.available()


def test_happy_group_photo():
    image = cv2.resize(cv2.imread("images/people.jpg"), None, fx=3, fy=3)
    moods = [FA.analyse(image, f).get("expression") for f in F.detect_faces(image)]
    assert moods.count("happy") >= 5          # a smiling team photo


def test_sentence_mentions_mood():
    image = cv2.resize(cv2.imread("images/people.jpg"), None, fx=5, fy=5)
    assert "happy" in F.recognize(image)[0]
    assert "مبسوط" in F.recognize(image, "ar")[0]


def test_age_words():
    assert FA.age_words("25-32") == "about 25 to 32 years old"
    assert FA.age_words("60+") == "over 60"
    assert FA.age_words(None) == ""


def test_votes_steady():
    v = FA.Votes()
    p = np.zeros(8); p[4] = 1
    for m in ["calm", "happy", "happy"]:
        v.add({"mood": m, "age_scores": p})
    assert v.mood == "happy" and v.sentiment == "positive" and v.age_group == "25-32"




