# preprocess.py
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import urllib.request
import os

# Download model if not present, if have skipped it,
MODEL_PATH = "blaze_face_short_range.tflite"
if not os.path.exists(MODEL_PATH):
    print("Downloading face detection model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        MODEL_PATH
    )

# Initialize new-style Face Detector
#configure model with a confidence threshold of 0.5 (50%)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)

#Detect face in a single image
def detect_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.detections:
        det = result.detections[0]
        bbox = det.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height

        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return (x1, y1, x2, y2)

    return None

#  Crop & resize face images
def crop_resize(image, box, target_size=(224, 224)):

    x1, y1, x2, y2 = box
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image.shape[1], x2 + pad_x)
    y2 = min(image.shape[0], y2 + pad_y)

    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face.astype("float32") / 255.0

#  Preprocess single image
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f" Image not found: {image_path}")
        return None

    box = detect_face(img)
    if box:
        face = crop_resize(img, box, target_size)
    else:
        print(f" No face detected: {Path(image_path).name}")
        face = cv2.resize(img, target_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype("float32") / 255.0

    return face

#  Preprocess entire dataset
def preprocess_dataset(image_dir, target_size=(224, 224), extensions=(".jpg", ".jpeg", ".png")):

    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f" Directory not found: {image_dir}")
        return np.array([]), []

    all_paths = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in extensions]

    if not all_paths:
        print(f" No images found in: {image_dir}")
        return np.array([]), []

    print(f" Found {len(all_paths)} images in '{image_dir}'")


    images, paths = [], []
    no_face_list = []
    not_found_list = []
    total = len(all_paths)

    for i, path in enumerate(all_paths, 1):
        img = cv2.imread(str(path))

        if img is None:
            not_found_list.append(path.name)
            print(f"  Image not readable: {path.name}")
            continue

        box = detect_face(img)

        if box is None:
            no_face_list.append(path.name)
            print(f" No face detected: {path.name}")

        face = preprocess_image(path, target_size)
        if face is not None:
            images.append(face)
            paths.append(str(path))

        if i % 50 == 0 or i == total:
            print(f" Processed {i}/{total} images — faces found so far: {len(images) - len(no_face_list)}")

    #  Final Summary
    print(" PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"  Total images found     : {total}")
    print(f"  Successfully processed : {len(images)}")
    print(f"  Faces detected         : {len(images) - len(no_face_list)}")
    print(f"  No face detected       : {len(no_face_list)}")
    print(f"  Unreadable images      : {len(not_found_list)}")

    if no_face_list:
        print(f"\n  Images with NO face detected ({len(no_face_list)}):")
        for name in no_face_list:
            print(f"     - {name}")

    if not_found_list:
        print(f"\n  Unreadable/missing images ({len(not_found_list)}):")
        for name in not_found_list:
            print(f"     - {name}")

    if not no_face_list and not not_found_list:
        print("\n All images processed successfully with faces detected!")

    if not images:
        return np.array([]), []

    return np.array(images, dtype="float32"), paths

# Run as script
if __name__ == "__main__":
    X, file_paths = preprocess_dataset("dataset/images")
    print(f"\nDone → array shape: {X.shape}, processed {len(file_paths)} images")
