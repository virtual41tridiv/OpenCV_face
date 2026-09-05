# Recognise faces from the webcam using the datasets built by
# face_data_collection.py, with a K-nearest-neighbours classifier.
#
# Run face_data_collection.py once per person first, then run this.

import os

import cv2
import numpy as np

dataset_path = './data/'
face_size = (100, 100)   # must match face_data_collection.py
offset = 10              # must match face_data_collection.py
k = 5                    # neighbours to vote
max_distance = 6000      # above this, call the face Unknown


def load_dataset(path):
    """Load every .npy in the dataset folder, one file per person."""
    faces, labels, names = [], [], {}

    if not os.path.isdir(path):
        return None, None, names

    for file_name in sorted(os.listdir(path)):
        if not file_name.endswith('.npy'):
            continue

        data = np.load(path + file_name)
        if data.size == 0:
            print("Skipping empty dataset: " + file_name)
            continue

        class_id = len(names)
        names[class_id] = file_name[:-4]
        faces.append(data)
        labels.append(np.full((data.shape[0],), class_id))
        print("Loaded " + str(data.shape[0]) + " faces for " + names[class_id])

    if not faces:
        return None, None, names

    return np.concatenate(faces).astype(np.float32), np.concatenate(labels), names


def knn(train_x, train_y, test, k=5):
    """Return the majority label among the k closest faces, and their mean distance."""
    distances = np.sqrt(((train_x - test) ** 2).sum(axis=1))
    nearest = np.argsort(distances)[:k]
    labels, counts = np.unique(train_y[nearest], return_counts=True)
    return labels[np.argmax(counts)], distances[nearest].mean()


train_x, train_y, names = load_dataset(dataset_path)

if train_x is None:
    print("No face data found in " + dataset_path)
    print("Run 'python face_data_collection.py' first to record at least one person.")
    raise SystemExit(1)

print("Ready: " + str(len(names)) + " people, " + str(train_x.shape[0]) + " total faces")

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop the same way face_data_collection.py did, but clamp to the frame
        # so a face at the edge does not produce an empty slice.
        y1, y2 = max(y - offset, 0), min(y + h + offset, frame.shape[0])
        x1, x2 = max(x - offset, 0), min(x + w + offset, frame.shape[1])
        face_section = frame[y1:y2, x1:x2]

        if face_section.size == 0:
            continue

        face_section = cv2.resize(face_section, face_size)

        label, distance = knn(train_x, train_y, face_section.flatten().astype(np.float32), k)

        if distance > max_distance:
            name, colour = "Unknown", (0, 0, 255)
        else:
            name, colour = names[int(label)], (0, 255, 255)

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, colour, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    cv2.imshow("Face Recognition", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
