# OpenCV_face

A small set of Python scripts that build up to real-time face recognition,
using OpenCV's Haar cascade classifiers and your webcam.

The project walks through five steps, each in its own script:

1. read and display a still image,
2. read a live stream from the webcam,
3. detect faces and eyes in that live stream,
4. crop the detected faces and save them to disk as a dataset,
5. recognise who is on camera by matching against those datasets.

## Requirements

- Python 3.8 or newer
- A webcam (the built-in FaceTime camera is fine)
- `opencv-python` **4.x** and `numpy`

> OpenCV 5.0 removed `CascadeClassifier` and the Haar cascade API altogether, so
> this project does not run on it. `requirements.txt` pins `opencv-python<5`.
> If you install OpenCV yourself, keep it on the 4.x series, or you will get
> `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`.

## Setup on macOS

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/virtual41tridiv/OpenCV_face.git
cd OpenCV_face
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use conda instead, this works too:

```bash
conda create -n opencv-face python=3.12
conda activate opencv-face
pip install -r requirements.txt
```

### Camera permission

The first time you run a script that opens the webcam, macOS asks whether the
app running Python may use the camera. The prompt goes to the **terminal app**
(Terminal, iTerm, VS Code…), not to Python itself.

If you never see the prompt, or you denied it by accident, enable it manually in
**System Settings → Privacy & Security → Camera** and restart the terminal.

Symptoms of a missing permission: a black window, or a stream of
`OpenCV: camera failed to properly initialize!` messages.

## Running the scripts

Run each one from the repo root, so the `haarcascade_*.xml` files are found:

```bash
python opencv.py path/to/photo.jpg
```

Opens a still image in a window. Press any key to close it.

```bash
python opencv_web.py
```

Opens the webcam and shows the raw video feed. Press `q` to quit.

```bash
python opencv_face.py
```

The main demo. Draws a blue box around each detected face and a green box
around each eye, and shows the grayscale frame alongside. Press `q` to quit.

```bash
python face_data_collection.py
```

Asks for a person's name, then records the largest face in frame. Every 10th
frame is cropped to 100×100 and kept. Press `q` to stop; the collected faces are
flattened and saved to `data/<name>.npy`.

Note that the window only refreshes while a face is actually detected, so if
nothing happens, move into better lighting and face the camera straight on.

```bash
python face_recognition.py
```

Loads every dataset in `data/`, then labels faces in the live stream with the
name of whoever they match. Press `q` to quit.

Faces that do not match anything closely enough are labelled `Unknown` in red
rather than being forced onto the nearest person.

> `q` must be pressed while an OpenCV **window** has focus, not the terminal. If a
> window ever refuses to close, `Ctrl+C` in the terminal will do it.

## Typical workflow

```bash
python face_data_collection.py    # enter "alice", move around, press q
python face_data_collection.py    # enter "bob",   move around, press q
python face_recognition.py        # both are now labelled on camera
```

Aim for at least 20–30 captured samples per person (the counter prints as it
collects) and vary your angle and expression a little, since the classifier only
knows what it has seen.

## Files

| File | What it does |
| --- | --- |
| `opencv.py` | Loads an image from a path and displays it |
| `opencv_web.py` | Streams the webcam to a window |
| `opencv_face.py` | Detects faces and eyes in the webcam stream |
| `face_data_collection.py` | Crops detected faces and saves them as a `.npy` dataset |
| `face_recognition.py` | Identifies faces in the live stream with a KNN classifier |
| `haarcascade_frontalface_alt.xml` | Pre-trained frontal-face cascade |
| `haarcascade_eye.xml` | Pre-trained eye cascade |

## How it works

Haar cascades are classic (pre-deep-learning) object detectors. Each cascade is
a chain of simple contrast-based filters trained on thousands of positive and
negative examples; a region has to pass every stage to count as a detection,
which makes rejecting the mostly-empty parts of a frame very cheap.

`detectMultiScale(image, 1.3, 5)` slides that detector over the image at many
scales. The two numbers are the knobs worth tuning:

- **`scaleFactor` (1.3)** — how much the search window shrinks between passes.
  Lower (e.g. `1.05`) finds more faces but runs slower.
- **`minNeighbors` (5)** — how many overlapping hits a region needs to be kept.
  Lower gives more detections and more false positives; higher is stricter.

Detection works on the grayscale frame, since the filters only care about
intensity, while the rectangles are drawn on the original color frame.

### Recognition

Detection finds *a* face; recognition works out *whose* it is. Each cropped face
is a 100×100 colour image, which flattened is just a list of 30,000 numbers — a
single point in 30,000-dimensional space. `face_recognition.py` loads every
saved point, and for a new face measures the straight-line distance to all of
them, takes the `k` (default 5) nearest, and lets them vote on the answer.

That is K-nearest-neighbours, and it has no training step at all: the dataset
*is* the model. It is the simplest thing that works, which makes it a good
teaching example and a poor production system — it compares against every stored
face on every frame, and it compares raw pixels, so lighting shifts move a face
further than a change of person can.

If the mean distance to those neighbours exceeds `max_distance` (6000), the face
is reported as `Unknown`. Without that check KNN always answers with *someone*,
since there is always a nearest point. Tune the threshold in the script if you
get the wrong answer: too many false `Unknown`s means raise it, strangers being
labelled with a name means lower it.

## Limitations

- Haar cascades only reliably detect **frontal** faces. Profile views, heavy
  tilt, glasses, or poor lighting will drop detections.
- Recognition compares raw pixels, so it is sensitive to lighting, distance and
  head angle. Capture your samples in roughly the conditions you will use it in.
- KNN scans the whole dataset for every face in every frame. That is fine for a
  few hundred samples and will crawl at tens of thousands.
- It cannot tell a photo from a person — holding up a picture fools it. A modern
  embedding model (OpenCV's `FaceRecognizerSF`, `face_recognition`, or any
  FaceNet-style network) is the upgrade path if you want robustness.

## License

No license specified — personal learning project.
