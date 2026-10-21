import os
import glob
import pickle
import cv2
import numpy as np
from deepface import DeepFace

DATASET_DIR = "dataset"
CACHE_FILE = "embeddings_cache.pkl"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
MODEL_NAME = "Facenet512"
SAMPLE_LIMIT_OWNER = 40
SAMPLE_LIMIT_OTHERS = 15

NAME_FORMAT_MAP = {
    "Joy manalu": "Joy Manalu",
    "citra": "Citra Yohana Sinaga",
    "dame": "Dame",
    "devi": "Devi Manurung",
    "gebyardi tato": "Gebyardi Tato Purba",
    "heelga pardede": "Helga Pardede",
    "jefry": "Jeffry Marbun",
    "josep simorangkir": "Josep Simorangkir",
    "keysa manik": "Keysa Manik",
    "lilis gultom": "Lilis Gultom",
    "mariansyah": "Mariansyah",
    "micael saragih": "Micael Saragih",
    "micael sinaga": "Micael Sinaga",
    "mutiara": "Mutiara",
    "putra": "Putra",
    "renold": "Renold",
    "reyhan": "Reyhan",
    "romarina": "Romarina Sinambela",
    "selviani": "Selviani Halawa",
    "vania nainggolan": "Vania Nainggolan",
    "viena siregar": "Viena Siregar",
    "zesryn damanik": "Zesryn Damanik",
}

def format_name(folder_name):
    return NAME_FORMAT_MAP.get(folder_name, folder_name.replace('_', ' ').title())

def align_face_landmarks(img_bgr, landmarks, box):
    """Aligns eyes horizontally based on 5 facial landmarks for maximum recognition accuracy."""
    re_x, re_y = landmarks[0]
    le_x, le_y = landmarks[1]
    dx = le_x - re_x
    dy = le_y - re_y
    angle = np.degrees(np.arctan2(dy, dx))
    
    center_x = (re_x + le_x) / 2.0
    center_y = (re_y + le_y) / 2.0
    
    w, h = box[2], box[3]
    size = int(max(w, h) * 1.35)
    size = max(64, min(size, 800))
    
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    M[0, 2] += size / 2.0 - center_x
    M[1, 2] += size / 2.0 - center_y
    
    aligned = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR)
    return aligned

def build_database():
    print(f"================================================================")
    print(f"  BUILDING HIGH-ACCURACY DATABASE: YuNet Aligned + {MODEL_NAME}")
    print(f"================================================================")
    
    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory '{DATASET_DIR}' not found!")
        return

    # Check detector
    if not os.path.exists(YUNET_MODEL):
        print(f"[ERROR] YuNet model '{YUNET_MODEL}' not found!")
        return

    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.6)

    person_dirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    person_dirs.sort()

    db = {
        "model_name": MODEL_NAME,
        "detector_backend": "yunet_aligned",
        "persons": {}
    }

    total_embedded = 0
    for folder in person_dirs:
        folder_path = os.path.join(DATASET_DIR, folder)
        image_files = glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]")) + \
                      glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]"))
        
        display_name = format_name(folder)
        is_owner = (folder.lower() == "joy manalu")
        sample_limit = SAMPLE_LIMIT_OWNER if is_owner else SAMPLE_LIMIT_OTHERS
        selected_files = image_files[:sample_limit]
        
        embeddings = []
        for img_path in selected_files:
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                h_img, w_img = img_bgr.shape[:2]
                detector.setInputSize((w_img, h_img))
                detected = detector.detect(img_bgr)
                
                if detected[1] is None or len(detected[1]) == 0:
                    # Fallback to direct crop if YuNet misses
                    crop = img_bgr
                else:
                    f0 = detected[1][0]
                    box = f0[0:4]
                    landmarks = f0[4:14].reshape(5, 2)
                    crop = align_face_landmarks(img_bgr, landmarks, box)

                rep = DeepFace.represent(
                    img_path=crop,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )
                if rep and len(rep) > 0 and 'embedding' in rep[0]:
                    emb = np.array(rep[0]['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    embeddings.append(emb)
            except Exception:
                continue

        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            
            db["persons"][folder] = {
                "display_name": display_name,
                "is_owner": is_owner,
                "photo_count": len(image_files),
                "embedded_count": len(embeddings),
                "embeddings": embeddings,
                "centroid": centroid
            }
            total_embedded += len(embeddings)
            print(f"  [OK] {display_name:<24} : {len(embeddings):>2}/{len(image_files):<2} aligned & embedded")
        else:
            print(f"  [WARN] {display_name:<24} : 0 embeddings")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(db, f)

    print(f"================================================================")
    print(f"  SUCCESS: {len(db['persons'])} identities, {total_embedded} total embeddings")
    print(f"  Saved to: '{CACHE_FILE}'")
    print(f"================================================================")

if __name__ == "__main__":
    build_database()
