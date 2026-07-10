"""
Sign Speak AI — Real-Time Gesture Data Collection Tool
=======================================================
Run this script from the project root:
    python ai_engine/data_collection.py

Controls:
  SPACE   → Capture the current hand landmark sample
  N       → Skip to the next gesture class
  Q       → Quit and save all collected data
  R       → Reset samples for the current gesture (start over)

Samples are saved to:  ai_engine/dataset/<ClassName>.npy
Each file contains a (num_samples, 63) float32 numpy array.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import time

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "ai_engine", "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ── Gesture classes to collect (must match train_model.py) ────────────────────
GESTURE_CLASSES = [
    "Hello",
    "Bye",
    "Good Morning",
    "Help",
    "I Love You",
    "No",
    "Sorry",
    "Thank You",
    "Thumbs Up",
    "Thumbs Down",
    "Welcome",
    "Yes",
]

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_SAMPLES   = 200        # samples to collect per gesture
COOLDOWN_FRAMES  = 8          # minimum frames between auto-captures (debounce)
AUTO_CAPTURE     = False       # set True to auto-capture when hand is stable

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── Helper: extract normalised 63-float landmark vector ──────────────────────
def extract_landmarks(hand_landmarks):
    """
    Extracts 21 landmarks, subtracts the wrist position (landmark 0),
    then min-max normalises so all values fall within [-1.0, 1.0].
    Returns a flat float32 numpy array of length 63.
    """
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append(lm.x - wrist_x)
        coords.append(lm.y - wrist_y)
        coords.append(lm.z - wrist_z)

    coords = np.array(coords, dtype=np.float32)
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords

# ── Helper: draw a styled HUD on the frame ───────────────────────────────────
def draw_hud(frame, gesture_name, collected, target, status_msg, colour):
    h, w = frame.shape[:2]

    # Dark translucent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 30), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    # Progress bar background
    bar_w = int(w * 0.8)
    bar_x = (w - bar_w) // 2
    cv2.rectangle(frame, (bar_x, 90), (bar_x + bar_w, 110), (40, 40, 60), -1)

    # Progress bar fill
    progress = min(collected / target, 1.0)
    fill_w = int(bar_w * progress)
    bar_colour = (0, 220, 120) if progress < 1.0 else (60, 220, 60)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, 90), (bar_x + fill_w, 110), bar_colour, -1)

    # Text overlays
    cv2.putText(frame, f"Gesture: {gesture_name}",
                (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{collected}/{target} samples",
                (12, 58), cv2.FONT_HERSHEY_DUPLEX, 0.65, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, status_msg,
                (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 1, cv2.LINE_AA)

    # Bottom controls hint
    hint = "SPACE=Capture  N=Next  R=Reset  Q=Quit"
    cv2.putText(frame, hint,
                (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    return frame

# ── Helper: load existing samples from disk ──────────────────────────────────
def load_existing(gesture_name):
    fname = os.path.join(DATASET_DIR, f"{gesture_name}.npy")
    if os.path.exists(fname):
        data = np.load(fname)
        print(f"  ✓ Loaded {len(data)} existing samples for '{gesture_name}'")
        return list(data)
    return []

# ── Helper: save samples to disk ─────────────────────────────────────────────
def save_samples(gesture_name, samples):
    if not samples:
        return
    arr = np.array(samples, dtype=np.float32)
    fname = os.path.join(DATASET_DIR, f"{gesture_name}.npy")
    np.save(fname, arr)
    print(f"  💾 Saved {len(samples)} samples → {fname}")

# ── Main collection loop ──────────────────────────────────────────────────────
def run_collection():
    print("\n" + "═" * 60)
    print("  Sign Speak AI — Gesture Data Collection")
    print("═" * 60)
    print(f"  Dataset directory : {DATASET_DIR}")
    print(f"  Target per class  : {TARGET_SAMPLES} samples")
    print(f"  Classes           : {len(GESTURE_CLASSES)}")
    print("═" * 60 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam. Check camera connection.")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        gesture_idx  = 0
        samples      = load_existing(GESTURE_CLASSES[gesture_idx])
        cooldown     = 0
        status_msg   = "Show your hand and press SPACE to capture."
        status_color = (200, 200, 200)

        print(f"► Starting with gesture: {GESTURE_CLASSES[gesture_idx]}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ ERROR: Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)   # mirror for natural feel
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            hand_detected  = False
            landmark_vec   = None

            if results.multi_hand_landmarks:
                hand_detected = True
                hl = results.multi_hand_landmarks[0]

                # Draw landmark skeleton
                mp_drawing.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                landmark_vec = extract_landmarks(hl)

            current_gesture = GESTURE_CLASSES[gesture_idx]
            collected_count = len(samples)

            # ── Keyboard handling ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            cooldown = max(0, cooldown - 1)

            if key == ord('q') or key == ord('Q'):
                # Quit — save current gesture and exit
                save_samples(current_gesture, samples)
                print("\n✅ Collection finished. Saved all data.")
                break

            elif key == ord('n') or key == ord('N'):
                # Next gesture — save current and advance
                save_samples(current_gesture, samples)
                if gesture_idx < len(GESTURE_CLASSES) - 1:
                    gesture_idx += 1
                    current_gesture = GESTURE_CLASSES[gesture_idx]
                    samples = load_existing(current_gesture)
                    status_msg   = "Show your hand and press SPACE to capture."
                    status_color = (200, 200, 200)
                    print(f"\n► Next gesture: {current_gesture}")
                else:
                    print("\n✅ All gestures collected!")
                    break

            elif key == ord('r') or key == ord('R'):
                # Reset current gesture
                samples      = []
                status_msg   = "Reset! Show hand and press SPACE."
                status_color = (60, 180, 255)
                print(f"  ↺ Reset samples for '{current_gesture}'")

            elif key == ord(' '):
                # Manual capture
                if hand_detected and landmark_vec is not None and cooldown == 0:
                    samples.append(landmark_vec)
                    cooldown     = COOLDOWN_FRAMES
                    collected_count = len(samples)
                    status_msg   = f"✓ Captured! {collected_count}/{TARGET_SAMPLES}"
                    status_color = (60, 220, 60)

                    # Auto-advance when target is met
                    if collected_count >= TARGET_SAMPLES:
                        save_samples(current_gesture, samples)
                        if gesture_idx < len(GESTURE_CLASSES) - 1:
                            gesture_idx += 1
                            current_gesture = GESTURE_CLASSES[gesture_idx]
                            samples = load_existing(current_gesture)
                            status_msg   = f"✅ Done! Next: {current_gesture}"
                            status_color = (60, 220, 60)
                            print(f"\n✅ Target reached. Next gesture: {current_gesture}")
                        else:
                            save_samples(current_gesture, samples)
                            print("\n✅ All gestures collected!")
                            break
                elif not hand_detected:
                    status_msg   = "⚠ No hand detected. Show your hand clearly."
                    status_color = (40, 120, 255)

            # ── Draw HUD ──────────────────────────────────────────────────
            frame = draw_hud(
                frame,
                current_gesture,
                len(samples),
                TARGET_SAMPLES,
                status_msg,
                status_color,
            )

            # Hand-not-detected warning overlay
            if not hand_detected:
                h, w = frame.shape[:2]
                cv2.putText(frame, "No hand detected",
                            (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (40, 80, 255), 2, cv2.LINE_AA)

            cv2.imshow("Sign Speak AI — Data Collection", frame)

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Collection Summary")
    print("═" * 60)
    total = 0
    for cls in GESTURE_CLASSES:
        fname = os.path.join(DATASET_DIR, f"{cls}.npy")
        if os.path.exists(fname):
            count = len(np.load(fname))
            status = "✓" if count >= TARGET_SAMPLES else f"⚠ only {count}"
            print(f"  {status:20s}  {cls}")
            total += count
        else:
            print(f"  ✗ missing            {cls}")
    print(f"\n  Total samples collected: {total}")
    print("═" * 60)
    print("\nNext step: run  python ai_engine/train_model.py\n")


if __name__ == "__main__":
    run_collection()
