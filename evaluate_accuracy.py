import os
import glob
import pickle
import json
import time
import cv2
import numpy as np
from deepface import DeepFace

CACHE_FILE = "embeddings_cache.pkl"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
DATASET_DIR = "dataset"
OUTPUT_REPORT = "calibration_report.json"

def align_face_landmarks(img_bgr, landmarks, box):
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

def run_evaluation():
    print("=" * 70)
    print("  EVALUASI AKURASI NYATA & KALIBRASI AMBANG (LEAVE-ONE-OUT BENCHMARK)")
    print("=" * 70)

    if not os.path.exists(CACHE_FILE):
        print(f"Error: {CACHE_FILE} not found!")
        return

    with open(CACHE_FILE, "rb") as f:
        db = pickle.load(f)

    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.6)

    # Collect unseen test photos from each person
    test_cases = [] # (img_path, true_folder, true_name)
    person_dirs = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

    for folder in person_dirs:
        folder_path = os.path.join(DATASET_DIR, folder)
        image_files = sorted(glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                             glob.glob(os.path.join(folder_path, "*.[jJ][pP][eE][gG]")) + \
                             glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")))
        
        # Take unseen test photos (photos with index >= 16 for classmates, >= 41 for Joy)
        start_idx = 41 if folder.lower() == "joy manalu" else 16
        unseen_photos = image_files[start_idx : start_idx + 5]
        
        display_name = db["persons"].get(folder, {}).get("display_name", folder)
        for p in unseen_photos:
            test_cases.append((p, folder, display_name))

    print(f"Total Foto Uji Unseen : {len(test_cases)} foto dari {len(person_dirs)} identitas")
    print("Mengevaluasi setiap foto...")

    raw_results = []
    t_start = time.time()

    for idx, (img_path, true_folder, true_name) in enumerate(test_cases):
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            h_img, w_img = img_bgr.shape[:2]
            detector.setInputSize((w_img, h_img))
            detected = detector.detect(img_bgr)

            if detected[1] is None or len(detected[1]) == 0:
                crop = img_bgr
            else:
                f0 = detected[1][0]
                box = f0[0:4]
                landmarks = f0[4:14].reshape(5, 2)
                crop = align_face_landmarks(img_bgr, landmarks, box)

            rep = DeepFace.represent(
                img_path=crop,
                model_name=db.get("model_name", "Facenet512"),
                detector_backend="skip",
                enforce_detection=False
            )

            if not rep or len(rep) == 0:
                continue

            q_emb = np.array(rep[0]["embedding"], dtype=np.float32)
            q_emb = q_emb / np.linalg.norm(q_emb)

            # Match against database using hybrid Centroid + KNN Min Sample
            all_scores = []
            for candidate_folder, cand_data in db["persons"].items():
                # 1. Centroid distance
                centroid = cand_data.get("centroid")
                d_centroid = 1.0 - float(np.dot(q_emb, centroid)) if centroid is not None else 1.0
                
                # 2. KNN Sample distances
                dists = [1.0 - float(np.dot(q_emb, s)) for s in cand_data.get("embeddings", [])]
                d_min = min(dists) if dists else 1.0
                d_top3 = float(np.mean(sorted(dists)[:3])) if len(dists) >= 3 else d_min

                # Hybrid metric: 0.65 * min_sample + 0.35 * top3
                hybrid_dist = 0.65 * d_min + 0.35 * d_top3
                all_scores.append((candidate_folder, cand_data["display_name"], hybrid_dist, d_min, d_centroid))

            all_scores.sort(key=lambda x: x[2]) # sort by hybrid distance ascending
            best_folder, best_name, best_dist, min_dist, centroid_dist = all_scores[0]
            is_correct = (best_folder == true_folder)

            raw_results.append({
                "true_folder": true_folder,
                "true_name": true_name,
                "pred_folder": best_folder,
                "pred_name": best_name,
                "is_correct": is_correct,
                "hybrid_dist": best_dist,
                "min_dist": min_dist,
                "centroid_dist": centroid_dist
            })

        except Exception as e:
            continue

    total_evaluated = len(raw_results)
    eval_time = time.time() - t_start

    # Test range of distance thresholds
    threshold_candidates = [0.30, 0.33, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
    calibration_table = []
    best_thresh = 0.42
    best_f1 = -1.0

    print("\n" + "=" * 70)
    print(f"{'Ambang':<8} | {'Benar (TP)':<11} | {'Salah (FP)':<11} | {'Unknown (FN)':<12} | {'Akurasi':<9} | {'F1-Score':<8}")
    print("-" * 70)

    for th in threshold_candidates:
        tp = sum(1 for r in raw_results if r["is_correct"] and r["hybrid_dist"] <= th)
        fp = sum(1 for r in raw_results if not r["is_correct"] and r["hybrid_dist"] <= th)
        fn = sum(1 for r in raw_results if r["hybrid_dist"] > th)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp / total_evaluated) * 100.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

        print(f"{th:<8.2f} | {tp:<11} | {fp:<11} | {fn:<12} | {acc:<8.1f}% | {f1:<8.3f}")
        calibration_table.append({
            "threshold": th,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "accuracy": round(acc, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1, 4)
        })

    # Top-1 Rank-1 Identification Rate (unconstrained)
    rank1_correct = sum(1 for r in raw_results if r["is_correct"])
    rank1_accuracy = (rank1_correct / total_evaluated) * 100.0

    print("=" * 70)
    print(f"Hasil Evaluasi Sebenarnya (Tanpa Manipulasi):")
    print(f"  • Total Foto Teruji        : {total_evaluated} foto")
    print(f"  • Rank-1 Match Benar       : {rank1_correct}/{total_evaluated} ({rank1_accuracy:.2f}%)")
    print(f"  • Ambang Terkalibrasi (Best): {best_thresh:.2f}")
    print(f"  • Waktu Evaluasi           : {eval_time:.1f} detik ({eval_time/total_evaluated*1000:.1f} ms/foto)")
    print("=" * 70)

    # Save to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tested": total_evaluated,
        "rank1_correct": rank1_correct,
        "rank1_accuracy_pct": round(rank1_accuracy, 2),
        "optimal_threshold": best_thresh,
        "calibrated_formula": f"max(0.0, min(99.5, (1.0 - (dist / {best_thresh * 1.25:.4f})) * 100.0))",
        "threshold_grid": calibration_table
    }

    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Laporan kalibrasi tersimpan di: {OUTPUT_REPORT}")

if __name__ == "__main__":
    run_evaluation()
