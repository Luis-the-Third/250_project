import cv2
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ── CONFIG ──────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_FILE = "known_embeddings.npy"  # no longer used, but kept if you still want to cache
MODEL_NAME      = "Facenet"               # or "VGG-Face", "ArcFace", etc.
DIST_THRESHOLD  = 0.40                    # cosine distance threshold
# ────────────────────────────────────────────────────────────────────

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ── FACE DETECTOR ───────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── ASCII “X” Printer ────────────────────────────────────────────────
def print_ascii_X(size=9):
    for i in range(size):
        line = "".join("X" if j == i or j == size-1-i else " "
                       for j in range(size))
        print(line)

# ── VIDEO LOOP ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    exit(1)

cv2.namedWindow("DeepFace Recognition")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5
    )

    # draw live boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2)

    cv2.imshow("DeepFace Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        # — capture new face and save to known_faces/ as before —
        name = input("Enter name for new face: ").strip()
        if not name:
            print("No name entered, cancelling.")
            continue

        ret2, frame2 = cap.read()
        if not ret2:
            print("Error grabbing frame.")
            continue

        gray2  = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(
            gray2, scaleFactor=1.1, minNeighbors=5
        )
        if len(faces2) != 1:
            print(f"Error: found {len(faces2)} faces, need exactly 1.")
            continue

        (x2, y2, w2, h2) = faces2[0]
        crop = frame2[y2:y2+h2, x2:x2+w2]

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fn  = f"{name}_{ts}.jpg"
        out_fp  = os.path.join(KNOWN_FACES_DIR, out_fn)
        cv2.imwrite(out_fp, crop)
        print(f"Saved {out_fp}")

    elif key == ord('a'):
        # — try to match first face via DeepFace.find against known_faces/ —
        if len(faces) == 0:
            print("No face detected on screen.")
            continue

        (xA, yA, wA, hA) = faces[0]
        face_crop = frame[yA:yA+hA, xA:xA+wA]

        try:
            dfs = DeepFace.find(
                face_crop,                      # first arg: numpy array OK
                db_path=KNOWN_FACES_DIR,
                model_name=MODEL_NAME,
                detector_backend='opencv',
                distance_metric='cosine',
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            print(f"DeepFace.find error: {e}")
            continue

        # DeepFace.find returns a list of DataFrames (one per detected face in input)
        df = dfs[0] if isinstance(dfs, list) else dfs

        if df.empty:
            print("\nNo match — printing X:\n")
            print_ascii_X()
        else:
            # we got at least one row → success
            identity_path = df.iloc[0]["identity"]
            matched_name  = os.path.splitext(os.path.basename(identity_path))[0]
            print(f"✅ Matched: {matched_name}")

cap.release()
cv2.destroyAllWindows()
