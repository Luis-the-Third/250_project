import cv2
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace
from deepface.commons import functions, distance as dst

# ── CONFIG ──────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
EMBEDDINGS_FILE = "known_embeddings.npy"
NAMES_FILE      = "known_names.npy"
MODEL_NAME      = "Facenet"      # you can also try "VGG-Face", "ArcFace", etc.
DIST_THRESHOLD  = 0.40           # cosine distance threshold for a match
# ────────────────────────────────────────────────────────────────────

# ensure known_faces dir exists
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ── LOAD OR BUILD EMBEDDINGS ────────────────────────────────────────
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(NAMES_FILE):
    known_embeddings = list(np.load(EMBEDDINGS_FILE, allow_pickle=True))
    known_names      = list(np.load(NAMES_FILE, allow_pickle=True))
    print(f"Loaded {len(known_names)} embeddings from disk")
else:
    known_embeddings = []
    known_names      = []
    print("Computing embeddings for images in known_faces/ …")
    for fn in os.listdir(KNOWN_FACES_DIR):
        if fn.startswith('.'): 
            continue
        name, _ = os.path.splitext(fn)
        img_path = os.path.join(KNOWN_FACES_DIR, fn)
        # preprocess_face returns a 4D tensor; we take the first one
        tensor = functions.preprocess_face(img_path, target_size=(160,160), enforce_detection=True)[0]
        rep = DeepFace.represent(tensor, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        known_embeddings.append(rep)
        known_names.append(name)
        print(f" • {name}")
    # cache to disk
    np.save(EMBEDDINGS_FILE, known_embeddings)
    np.save(NAMES_FILE,      known_names)
    print("Done.")

# ── FACE DETECTOR ───────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # extract face ROI
        face_img = frame[y:y+h, x:x+w]
        try:
            tensor = functions.preprocess_face(face_img, target_size=(160,160), enforce_detection=False)[0]
            rep = DeepFace.represent(tensor, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        except Exception as e:
            # in case preprocessing fails
            continue

        # compare to known embeddings
        dists = [ dst.findCosineDistance(rep, ke) for ke in known_embeddings ]
        if len(dists):
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            name = known_names[best_idx] if best_dist < DIST_THRESHOLD else "Unknown"
        else:
            name, best_dist = "Unknown", None

        # draw box + label
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        label = f"{name}" + (f" {best_dist:.2f}" if best_dist is not None else "")
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("DeepFace Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        # CAPTURE FLOW
        name = input("Enter name for new face: ").strip()
        if not name:
            print("No name entered, cancelling.")
            continue

        print(f"Capturing face for “{name}”… please look at camera.")
        ret2, frame2 = cap.read()
        if not ret2:
            print("Error grabbing frame.")
            continue

        # detect & crop
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
        if len(faces2) != 1:
            print(f"Error: found {len(faces2)} faces, need exactly 1.")
            continue

        (x2, y2, w2, h2) = faces2[0]
        face_crop = frame2[y2:y2+h2, x2:x2+w2]

        # save image
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fn = f"{name}_{ts}.jpg"
        out_path = os.path.join(KNOWN_FACES_DIR, out_fn)
        cv2.imwrite(out_path, face_crop)
        print(f"Saved {out_path}")

        # compute embedding & append
        tensor = functions.preprocess_face(face_crop, target_size=(160,160), enforce_detection=False)[0]
        rep = DeepFace.represent(tensor, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        known_embeddings.append(rep)
        known_names.append(name)

        # update cache
        np.save(EMBEDDINGS_FILE, known_embeddings)
        np.save(NAMES_FILE,      known_names)
        print(f"Added {name} to known faces.")

cap.release()
cv2.destroyAllWindows()
