# OpenCV_face

A small set of Python scripts for playing with real-time face and eye detection
using OpenCV's Haar cascade classifiers and your webcam.

The project walks through four steps, each in its own script:

1. read and display a still image,
2. read a live stream from the webcam,
3. detect faces and eyes in that live stream,
4. crop the detected faces and save them to disk as a dataset.

## Requirements

- Python 3.8 or newer
- A webcam (the built-in FaceTime camera is fine)
- `opencv-python` and `numpy`

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

> `q` must be pressed while an OpenCV **window** has focus, not the terminal. If a
> window ever refuses to close, `Ctrl+C` in the terminal will do it.

## Files

| File | What it does |
| --- | --- |
| `opencv.py` | Loads an image from a path and displays it |
| `opencv_web.py` | Streams the webcam to a window |
| `opencv_face.py` | Detects faces and eyes in the webcam stream |
| `face_data_collection.py` | Crops detected faces and saves them as a `.npy` dataset |
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

## Limitations

- Haar cascades only reliably detect **frontal** faces. Profile views, heavy
  tilt, glasses, or poor lighting will drop detections.
- `face_data_collection.py` collects data but nothing in this repo classifies it
  yet — pairing it with a K-nearest-neighbours classifier over the saved `.npy`
  files is the natural next step.
- Faces very close to the frame edge can fail to crop, because the script pads
  the region by 10 pixels before resizing.

## License

No license specified — personal learning project.
