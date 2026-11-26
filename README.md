# Ru'ya (رؤية) — Smart Glasses for Blind and Visually Impaired People

*Ru'ya* means "vision" in Arabic. Formerly **Simulated Smart Glasses**.

Overview
This project, developed as a final year project for a Bachelor of Science in Artificial Intelligence at the Arab Open University - Egypt Branch, presents a software-based prototype simulating smart glasses to assist visually impaired individuals in navigating their environment independently. By leveraging existing devices like smartphones and webcams, the system provides real-time auditory feedback without requiring specialized hardware, enhancing accessibility and affordability. The project was completed by Ahmed Yasser Kamal Shaaban under the supervision of Dr. Ramdan Fawzy Mohamed Babers, achieving a comprehensive solution that integrates advanced computer vision and natural language processing technologies.

Features
Real-Time Object Detection: Utilizes YOLOv8 to detect and classify ~80 object classes, providing positional feedback (e.g., left, right, front) and distance estimation using real-world object dimensions.
Text Recognition (OCR): Employs Tesseract OCR to extract and read text aloud from images or documents using text-to-speech (pyttsx3), supporting printed text in English and Arabic.
Currency Recognition: Identifies Egyptian banknotes, aiding users in financial transactions, with a custom-trained YOLO model for accuracy.
Scene Description: Uses the BLIP model to generate natural language descriptions of environments, enhancing contextual awareness (e.g., "A chair is in front of you, a table to your right").
Voice Control: Supports hands-free operation via voice commands to switch modes (e.g., object detection, text reading, currency recognition) using the SpeechRecognition API.
Distance Estimation: Estimates object distances based on known dimensions, assisting with navigation and obstacle avoidance.
Face Recognition: Recognises people the user has added (family, friends) and announces who is nearby and where (e.g., "Mona is on your left, close to you"). Uses OpenCV's YuNet face detector and SFace face recogniser; faces are double-checked against YOLOv8 person detections to reject animal faces. Unknown people are reported as "an unknown person" and never stored.
System Architecture
Frontend: Built with HTML, CSS, and JavaScript (Bootstrap for responsiveness), offering a user-friendly web interface with webcam integration and voice command support. The interface is designed for accessibility, with high-contrast colors and clear auditory feedback.
Backend: Developed using Flask (Python), integrating YOLOv8, Tesseract OCR, and BLIP models for processing and feedback generation. The backend handles real-time image processing and voice command interpretation, ensuring minimal latency.
Data Flow: Users upload images or stream webcam feeds, processed by the backend, with results delivered as auditory feedback via text-to-speech. The system follows a modular design, with clear separation of frontend and backend tasks, as detailed in the project’s use case and class diagrams.
Technologies Used
Programming Languages: Python, JavaScript
Frameworks/Libraries:
Flask: Web framework for backend
YOLOv8: Object detection and classification
Tesseract OCR: Text recognition
BLIP: Scene description
pyttsx3: Text-to-speech
SpeechRecognition API: Voice control
OpenCV: Image processing
num2words: Converts numeric counts to words for natural language output
Tools: Draw.io (diagrams), Bootstrap (frontend responsiveness), Jinja2 (Flask templating for dynamic content)
Development Methodology
The project followed the Waterfall Model SDLC, ensuring a structured approach:

Requirement Gathering: Identified functional (e.g., object detection, OCR) and non-functional (e.g., performance, accessibility) requirements.
System Design: Created use case, class, activity, and sequence diagrams to define the system architecture.
Implementation: Integrated YOLOv8, Tesseract, BLIP, and voice control, with Flask handling backend logic.
Testing: Conducted white box, black box, unit, system, and user acceptance testing.
Maintenance: Planned for bug fixes and feature enhancements based on user feedback.
Testing
The system underwent rigorous testing to ensure reliability and usability:

White Box Testing: Validated internal logic of YOLOv8, Tesseract, and currency models, ensuring accurate outputs.
Black Box Testing: Tested user interactions, voice commands, and overall system performance from an end-user perspective.
Unit Testing: Verified individual components (e.g., object detection, OCR, voice recognition) with test cases.
System Testing: Ensured seamless integration of frontend, backend, and models across various scenarios.
User Acceptance Testing: Conducted with visually impaired users to refine voice control, interface, and feedback clarity.
Challenges and Limitations
Lighting Conditions: Object detection and OCR accuracy decrease in low-light or cluttered environments. Mitigated by optimizing image preprocessing but requires further model training.
Currency Recognition: Struggles with worn or damaged notes, requiring a more robust model.
Voice Recognition: Less accurate in noisy environments, needing enhanced noise filtering.
Performance: Real-time processing can lag with high-resolution images. Optimized by adjusting resolution, but further improvements are needed.
Small Objects: Distance estimation is less accurate for small or distant objects due to camera limitations.
Future Work
Multilingual Support: Extend OCR and text-to-speech to support additional languages (e.g., Spanish, German) for global accessibility.
Improved Object Detection: Train YOLOv8 on diverse datasets to enhance accuracy in complex scenes, including small or occluded objects.
Hardware Integration: Pair with AR glasses for a seamless, immersive experience, reducing reliance on external devices.
Advanced Scene Understanding: Incorporate spatial relationships and contextual insights (e.g., "a person is sitting on a bench") for richer descriptions.
Real-Time Navigation: Integrate GPS or indoor mapping for turn-by-turn guidance and obstacle avoidance.
User Personalization: Allow users to customize detection sensitivity, speech speed, or prioritized objects.
Ethical Considerations
Privacy: No personal data is stored; images are processed locally or on the server without retention, adhering to data privacy regulations.
Accessibility: Designed for ease of use by visually impaired individuals, with voice control, high-contrast UI, and auditory feedback.
Inclusivity: Promotes independence and social inclusion by leveraging affordable, widely available devices.
Ethical Development: Involved visually impaired users in testing to ensure the system meets their needs, with a commitment to continuous improvement.
Face Recognition Voice Commands
English: "who" (who is in front of me), "remember <name>" (add the person in front of the camera), "forget <name>", "who do you know".
Arabic: "مين", "احفظ <الاسم>", "انسى <الاسم>", "تعرف مين".
Add 3-5 photos per person from different angles for best results. Photos are stored locally in the faces/ folder, which is excluded from Git.


---
## Version 2 (November 2026): audit, refactor and new features

### New features
- **Face recognition** (YuNet + SFace) with **guided enrollment**: say "remember Ahmed" and the glasses take 3 photos ("look straight… turn a little left… a little right…").
- **Clock-face directions + distance in metres**: "The bottle is at 1 o'clock, about 1 meter away." This is the standard used in orientation & mobility training for blind people.
- **Guide mode (motion-aware)**: watches continuously and speaks only when something changes: obstacles very close ahead, someone approaching, a known friend appearing, a newcomer. Say "guide" / "stop guide" (Arabic: "إرشاد" / "وقف الإرشاد"), or press **G**.
- **Live boxes on the camera** for sighted people watching (names, distance), animated.
- **Offline Arabic**: the app writes its own Arabic sentences (no translation website). The whole interface switches to Arabic too.
- **Voice-first, hands-free**: a "Put on the glasses" start screen, sound cues (listening / understood / done / warning), "help", "repeat", "stop".

### Fixes from the audit
| # | Problem | Fix |
|---|---------|-----|
| 1 | Find crashed for TV, couch, potted plant, dining table, motorcycle, airplane (old class names) | New size table for all 80 YOLO classes + test that every class works |
| 2 | Description model got BGR colours and captions were cut/repeated | RGB input, beam search, repetition guard, junk-word cleanup |
| 3 | `debug=True` on `0.0.0.0` exposed the Werkzeug debugger to the network | Debug off (+ test) |
| 4 | Distance used a fixed focal length and feet | Field-of-view model, metres |
| 5 | Find only matched exact words | "Bottle", "bottles", "the bottle", synonyms, Arabic words |
| 6 | YOLOv8m loaded 3 times | One shared model registry (`core/models.py`) |
| 7 | Summariser (1.2 GB) reloaded on every request | Loaded once; short texts read as-is |
| 8 | OCR read random symbols from photos without text | Confidence filter, 2-pass preprocessing (Otsu + adaptive for glare), Arabic when installed |
| 9 | Detect didn't say where things are | Clock direction + distance for every object, nearest first |
| 10 | Every request saved an image to `runs/` | Removed (today's frames only, in `uploads/`) |
| 11 | Face recognition gave wrong names on look-alike scores | Margin rule + "might be" band (wrong identities 2 → 0 on 65 real faces) |
| 12 | Arabic depended on an online translator | Built-in Arabic sentences |

### Face recognition accuracy (65 real faces from the user's own camera, low light, hoods, glasses)
| | Before | After |
|---|---|---|
| Known faces named correctly | 47/55 | 48/55 (+2 "might be") — 46/51 (90%) with 3-photo enrollment |
| Wrong identity given | 2 | **0** |
| Stranger correctly "unknown" | 12/13 | 12/13 — 13/13 with 3-photo enrollment |

### Project structure
```
app.py                       Flask routes (/detect, /face/enroll, /guide, /health)
core/models.py               every AI model loaded once, shared, thread-safe
core/vision.py               clock directions, distance estimation, object sizes
core/i18n.py                 English / Arabic sentences and object names
detection_function.py        objects + currency (with totals)
finder_and_depth_estimation.py  find an object
face_function.py             face recognition + guided enrollment
guide.py                     Guide mode (tracking, approaching, obstacles)
describe_function_2.py       BLIP scene description
ocr_function.py              text reading
summarize_function.py        summary of text
templates/index.html         the interface
tests/                       52 unit + API tests (pytest)
```

### Tests
- `.venv\Scripts\python -m pytest tests` runs 53 unit/API tests.
- 20 end-to-end Playwright tests ran a real browser with a fake camera (real footage), a fake microphone speaking commands and a recorder of everything the glasses said: start, help, guided enrollment, recognition with boxes, two people, stranger never named, forget, detect, find, currency, Guide mode (approaching + known face), Arabic, hands-free, no camera, phone layout — **20/20 passed**.

### Demo script (3 minutes)
1. Tap **Put on the glasses** → it says "Smart glasses ready".
2. Say **"remember Ahmed"** → follow the 3 prompts.
3. Say **"who"** → "Ahmed is at 12 o'clock, about 1 meter away" (box + name on screen).
4. Ask someone not saved to step in → "an unknown person…".
5. Hold a bottle → **"find the bottle"** → clock direction + distance.
6. Say **"guide"** → walk towards the camera → "Person approaching…"; a saved friend → "Ahmed is at…".
7. Switch to **عربي** → **"مين"** → the answer in Arabic, interface in Arabic.

---
## Version 2.1 (24 November 2026)
- **Find anything** with an open-vocabulary model (YOLOE, `models/glasses_vocab.pt`): 63 everyday objects that the 80 standard classes miss: door, stairs, keys, glasses, wallet, medicine, socket, light switch, door handle, shoes, cane… in English and Arabic ("find my keys", "ابحث عن النضارة"). The vocabulary is stored inside the model file, so no internet or text model is needed.
- **Guide mode hazards**: stairs, escalators, poles, crossings and wet-floor signs are announced as they come into view.
- **Arabic text reading**: Tesseract Arabic + English data ship in `tessdata/` (no admin install). Arabic text is read with the Arabic voice even in English mode.
- **Microphone fix**: phone-camera apps (Iriun) install a silent virtual microphone and make it the Windows default. The page now listens to the real laptop microphone and warns if the default is virtual.
- **Camera fix**: a hidden tab releases the camera, a busy camera is retried automatically, and the chosen camera is remembered (phone camera preferred over "OBS Virtual Camera").
- **New visual language**: tactile-paving yellow on asphalt grey, Atkinson Hyperlegible (designed by the Braille Institute for low-vision readers) + IBM Plex Sans Arabic, a braille-cell logo. Fonts are bundled for offline use (OFL licence).
- **Currency retraining kit** (`training/`): merges Roboflow datasets, maps their class names to one list that adds the **new polymer 20-pound note**, adds no-money background photos to cut false alarms, trains on the NVIDIA GPU, and installs the new model only if it beats the current one on the same validation images (the old model is kept as `models/currency_old.pt`).
- Tests: 56 unit/API tests (`python -m pytest tests`).

## Version 2.2 (26 November 2026): mood, age group and distance in steps
- **Distance in steps** (new default): a blind user counts steps, not metres. "The door is at 12 o'clock, about 3 steps away" / "على بعد حوالي 3 خطوات". Closer than 60 cm: "within reach of your hand".
  - **My step** setting (default 65 cm; about 41% of body height), saved in the browser. Switch with the *Distance in: Steps / Metres* control or by voice: "in steps", "in metres", "my step is 70" / "بالخطوات"، "بالمتر"، "خطوتي 70".
  - The on-screen boxes show the same unit as the voice.
- **Mood**: OpenCV Zoo facial-expression model (Apache-2.0, 88.3% on RAF-DB). "Ahmed is at 12 o'clock, about 2 steps away, and looks happy". Calm faces are not announced. Groups: "Eight unknown people, most of them look happy". Tested: 7 of 8 smiling people in a team photo were read as happy.
- **Age group** for unknown people: GoogLeNet trained on Adience (8 groups). Published accuracy is ~50% for the exact group and ~85% within one neighbouring group, so it is said as "about 25 to 32" and only for faces at least 60 px wide. Adience images were collected for research: check the licence before any commercial use.
- **Read text / Summarise fix on Windows**: the Tesseract language folder is now set with `TESSDATA_PREFIX` (the old `--tessdata-dir` flag kept its quotes on Windows and failed).

## Related project
The CCTV and security work that started here (face lists, alarms, Egyptian plate reading, working time) became a separate project: [**Raqib (رقيب)**](https://github.com/Ahmed-Yasser-V0/Raqib).
