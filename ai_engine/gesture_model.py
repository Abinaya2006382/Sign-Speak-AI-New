import os
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque, Counter

# ── Resolve model paths ───────────────────────────────────────────────────────
MODEL_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model'))
MODEL_PATH   = os.path.join(MODEL_DIR, 'gesture_classifier.keras')
DATASET_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset'))

# ── Real-time prediction configuration ───────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.75   # ignore predictions below this confidence
PREDICTION_BUFFER_SIZE = 10     # number of recent frames to keep in sliding window
MAJORITY_VOTE_MIN      = 5      # minimum votes in buffer to confirm a gesture


class GestureClassifier:
    def __init__(self):
        """Initialises the classifier. Loads the model if available, otherwise sets up to build/train."""
        self.model   = None
        self.classes = []
        os.makedirs(MODEL_DIR, exist_ok=True)

        # ── Sliding-window prediction buffer ─────────────────────────────────
        # Stores (class_index, confidence) pairs for the last N confident frames.
        self._pred_buffer: deque = deque(maxlen=PREDICTION_BUFFER_SIZE)

    # ─────────────────────────────────────────────────────────────────────────
    # SYNTHETIC DATA (kept for backward-compatibility with the retrain API)
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_synthetic_landmarks(self, gesture_name, num_samples=1000):
        """
        Generates synthetic hand landmarks (63 floats per sample) with Gaussian noise,
        random 3D rotation, and scaling based on structural templates.

        NOTE: This method is kept for backward-compatibility only. Real data from
        ai_engine/dataset/ is always preferred by train_model() when available.
        """
        # Base templates (21 landmarks * 3 coords = 63 values)
        base_hand = np.zeros((21, 3))

        # Setup generic hand skeletal structure (wrist at 0,0,0)
        base_hand[1]  = [0.1,   -0.15, -0.05]   # Thumb base
        base_hand[5]  = [0.08,  -0.3,  -0.05]   # Index knuckle
        base_hand[9]  = [0.0,   -0.35, -0.05]   # Middle knuckle
        base_hand[13] = [-0.08, -0.3,  -0.05]   # Ring knuckle
        base_hand[17] = [-0.15, -0.25, -0.05]   # Pinky knuckle

        gesture_key = gesture_name.lower().replace('_', ' ').strip()

        # ── TEMPLATE CREATION ─────────────────────────────────────────────────
        if 'hello' in gesture_key or 'hi' in gesture_key:
            # HELLO: All fingers fully extended — open-palm wave
            base_hand[2]  = [0.18, -0.25, -0.08]; base_hand[3]  = [0.24, -0.32, -0.10]; base_hand[4]  = [0.28, -0.38, -0.12]
            base_hand[6]  = [0.08, -0.45, -0.07]; base_hand[7]  = [0.08, -0.60, -0.09]; base_hand[8]  = [0.08, -0.72, -0.11]
            base_hand[10] = [0.00, -0.50, -0.07]; base_hand[11] = [0.00, -0.66, -0.09]; base_hand[12] = [0.00, -0.80, -0.11]
            base_hand[14] = [-0.08,-0.45, -0.07]; base_hand[15] = [-0.08,-0.60, -0.09]; base_hand[16] = [-0.08,-0.72, -0.11]
            base_hand[18] = [-0.15,-0.38, -0.07]; base_hand[19] = [-0.15,-0.50, -0.09]; base_hand[20] = [-0.15,-0.60, -0.11]

        elif 'bye' in gesture_key:
            # BYE: All fingers extended but hand rotated slightly (wrist tilt to differentiate from Hello)
            base_hand[2]  = [0.20, -0.22, -0.06]; base_hand[3]  = [0.27, -0.28, -0.08]; base_hand[4]  = [0.32, -0.34, -0.10]
            base_hand[6]  = [0.10, -0.42, -0.06]; base_hand[7]  = [0.10, -0.58, -0.08]; base_hand[8]  = [0.10, -0.70, -0.10]
            base_hand[10] = [0.02, -0.47, -0.06]; base_hand[11] = [0.02, -0.63, -0.08]; base_hand[12] = [0.02, -0.77, -0.10]
            base_hand[14] = [-0.06,-0.42, -0.06]; base_hand[15] = [-0.06,-0.58, -0.08]; base_hand[16] = [-0.06,-0.70, -0.10]
            base_hand[18] = [-0.13,-0.35, -0.06]; base_hand[19] = [-0.13,-0.47, -0.08]; base_hand[20] = [-0.13,-0.57, -0.10]

        elif 'yes' in gesture_key:
            # YES: Fist bobbing — all fingers tightly curled, thumb side
            base_hand[2]  = [0.12, -0.18, -0.04]; base_hand[3]  = [0.08, -0.20, -0.04]; base_hand[4]  = [0.04, -0.20, -0.04]
            base_hand[6]  = [0.09, -0.22, -0.03]; base_hand[7]  = [0.07, -0.18, -0.02]; base_hand[8]  = [0.05, -0.15, -0.01]
            base_hand[10] = [0.00, -0.25, -0.03]; base_hand[11] = [0.00, -0.20, -0.02]; base_hand[12] = [0.00, -0.16, -0.01]
            base_hand[14] = [-0.07,-0.22, -0.03]; base_hand[15] = [-0.07,-0.18, -0.02]; base_hand[16] = [-0.07,-0.15, -0.01]
            base_hand[18] = [-0.13,-0.18, -0.03]; base_hand[19] = [-0.13,-0.15, -0.02]; base_hand[20] = [-0.13,-0.12, -0.01]

        elif 'sorry' in gesture_key:
            # SORRY: Closed fist with thumb crossing knuckles (circular chest motion context)
            base_hand[2]  = [0.05, -0.15, -0.02]; base_hand[3]  = [0.00, -0.18, -0.03]; base_hand[4]  = [-0.05,-0.20, -0.04]
            base_hand[6]  = [0.09, -0.22, -0.03]; base_hand[7]  = [0.07, -0.18, -0.02]; base_hand[8]  = [0.05, -0.15, -0.01]
            base_hand[10] = [0.00, -0.25, -0.03]; base_hand[11] = [0.00, -0.20, -0.02]; base_hand[12] = [0.00, -0.16, -0.01]
            base_hand[14] = [-0.07,-0.22, -0.03]; base_hand[15] = [-0.07,-0.18, -0.02]; base_hand[16] = [-0.07,-0.15, -0.01]
            base_hand[18] = [-0.13,-0.18, -0.03]; base_hand[19] = [-0.13,-0.15, -0.02]; base_hand[20] = [-0.13,-0.12, -0.01]

        elif 'thank' in gesture_key:
            # THANK YOU: Flat open hand, fingers together, palm forward (NOT same as Help)
            base_hand[2]  = [0.08, -0.22, -0.04]; base_hand[3]  = [0.10, -0.30, -0.06]; base_hand[4]  = [0.11, -0.36, -0.08]
            base_hand[6]  = [0.06, -0.45, -0.12]; base_hand[7]  = [0.06, -0.58, -0.18]; base_hand[8]  = [0.06, -0.68, -0.24]
            base_hand[10] = [0.01, -0.48, -0.12]; base_hand[11] = [0.01, -0.61, -0.18]; base_hand[12] = [0.01, -0.71, -0.24]
            base_hand[14] = [-0.04,-0.45, -0.12]; base_hand[15] = [-0.04,-0.58, -0.18]; base_hand[16] = [-0.04,-0.68, -0.24]
            base_hand[18] = [-0.09,-0.38, -0.10]; base_hand[19] = [-0.09,-0.49, -0.16]; base_hand[20] = [-0.09,-0.57, -0.22]

        elif 'help' in gesture_key:
            # HELP: Flat hand moving upward under other (ASL) — palm-up with slight forward tilt
            base_hand[2]  = [0.15, -0.20, -0.06]; base_hand[3]  = [0.22, -0.26, -0.08]; base_hand[4]  = [0.26, -0.30, -0.09]
            base_hand[6]  = [0.08, -0.40, -0.10]; base_hand[7]  = [0.08, -0.52, -0.15]; base_hand[8]  = [0.08, -0.62, -0.20]
            base_hand[10] = [0.00, -0.44, -0.10]; base_hand[11] = [0.00, -0.56, -0.15]; base_hand[12] = [0.00, -0.66, -0.20]
            base_hand[14] = [-0.08,-0.40, -0.10]; base_hand[15] = [-0.08,-0.52, -0.15]; base_hand[16] = [-0.08,-0.62, -0.20]
            base_hand[18] = [-0.15,-0.34, -0.10]; base_hand[19] = [-0.15,-0.44, -0.15]; base_hand[20] = [-0.15,-0.52, -0.20]

        elif 'love' in gesture_key or 'ily' in gesture_key:
            # I LOVE YOU: Thumb + Index + Pinky extended; Middle + Ring curled
            base_hand[2]  = [0.18, -0.22, -0.06]; base_hand[3]  = [0.25, -0.28, -0.08]; base_hand[4]  = [0.32, -0.32, -0.10]
            base_hand[6]  = [0.08, -0.45, -0.07]; base_hand[7]  = [0.08, -0.60, -0.09]; base_hand[8]  = [0.08, -0.72, -0.11]
            base_hand[10] = [0.00, -0.25, -0.03]; base_hand[11] = [0.00, -0.20, -0.02]; base_hand[12] = [0.00, -0.16, -0.01]
            base_hand[14] = [-0.07,-0.22, -0.03]; base_hand[15] = [-0.07,-0.18, -0.02]; base_hand[16] = [-0.07,-0.15, -0.01]
            base_hand[18] = [-0.15,-0.38, -0.07]; base_hand[19] = [-0.18,-0.50, -0.09]; base_hand[20] = [-0.22,-0.62, -0.11]

        elif 'thumbs up' in gesture_key:
            # THUMBS UP: Fist with ONLY thumb pointing upward (negative Y = up on screen)
            base_hand[2]  = [0.12, -0.28, -0.05]; base_hand[3]  = [0.16, -0.44, -0.08]; base_hand[4]  = [0.20, -0.60, -0.10]
            # Other four fingers tightly curled
            base_hand[6]  = [0.10, -0.22, -0.03]; base_hand[7]  = [0.08, -0.18, -0.02]; base_hand[8]  = [0.06, -0.15, -0.01]
            base_hand[10] = [0.00, -0.22, -0.03]; base_hand[11] = [0.00, -0.18, -0.02]; base_hand[12] = [0.00, -0.15, -0.01]
            base_hand[14] = [-0.08,-0.22, -0.03]; base_hand[15] = [-0.08,-0.18, -0.02]; base_hand[16] = [-0.08,-0.15, -0.01]
            base_hand[18] = [-0.14,-0.18, -0.03]; base_hand[19] = [-0.14,-0.15, -0.02]; base_hand[20] = [-0.14,-0.12, -0.01]

        elif 'thumbs down' in gesture_key:
            # THUMBS DOWN: Fist with ONLY thumb pointing downward (positive Y = down on screen)
            base_hand[2]  = [0.12, -0.10, -0.05]; base_hand[3]  = [0.16,  0.06, -0.08]; base_hand[4]  = [0.20,  0.22, -0.10]
            # Other four fingers tightly curled — IDENTICAL to Thumbs Up body (key difference is thumb direction)
            base_hand[6]  = [0.10, -0.22, -0.03]; base_hand[7]  = [0.08, -0.18, -0.02]; base_hand[8]  = [0.06, -0.15, -0.01]
            base_hand[10] = [0.00, -0.22, -0.03]; base_hand[11] = [0.00, -0.18, -0.02]; base_hand[12] = [0.00, -0.15, -0.01]
            base_hand[14] = [-0.08,-0.22, -0.03]; base_hand[15] = [-0.08,-0.18, -0.02]; base_hand[16] = [-0.08,-0.15, -0.01]
            base_hand[18] = [-0.14,-0.18, -0.03]; base_hand[19] = [-0.14,-0.15, -0.02]; base_hand[20] = [-0.14,-0.12, -0.01]

        elif 'welcome' in gesture_key:
            # WELCOME: Open hand, fingers relaxed and slightly curved inward
            base_hand[2]  = [0.15, -0.20, -0.06]; base_hand[3]  = [0.22, -0.26, -0.08]; base_hand[4]  = [0.28, -0.30, -0.09]
            base_hand[6]  = [0.08, -0.38, -0.08]; base_hand[7]  = [0.08, -0.48, -0.12]; base_hand[8]  = [0.08, -0.56, -0.15]
            base_hand[10] = [0.00, -0.42, -0.08]; base_hand[11] = [0.00, -0.52, -0.12]; base_hand[12] = [0.00, -0.60, -0.15]
            base_hand[14] = [-0.08,-0.38, -0.08]; base_hand[15] = [-0.08,-0.48, -0.12]; base_hand[16] = [-0.08,-0.56, -0.15]
            base_hand[18] = [-0.15,-0.32, -0.08]; base_hand[19] = [-0.15,-0.40, -0.12]; base_hand[20] = [-0.15,-0.48, -0.15]

        elif 'good morning' in gesture_key:
            # GOOD MORNING: Flat hand, fingers closed together, palm outward
            base_hand[2]  = [0.08, -0.22, -0.04]; base_hand[3]  = [0.10, -0.32, -0.06]; base_hand[4]  = [0.11, -0.40, -0.08]
            base_hand[6]  = [0.05, -0.45, -0.07]; base_hand[7]  = [0.05, -0.60, -0.09]; base_hand[8]  = [0.05, -0.72, -0.11]
            base_hand[10] = [0.01, -0.48, -0.07]; base_hand[11] = [0.01, -0.64, -0.09]; base_hand[12] = [0.01, -0.78, -0.11]
            base_hand[14] = [-0.03,-0.45, -0.07]; base_hand[15] = [-0.03,-0.60, -0.09]; base_hand[16] = [-0.03,-0.72, -0.11]
            base_hand[18] = [-0.07,-0.38, -0.07]; base_hand[19] = [-0.07,-0.50, -0.09]; base_hand[20] = [-0.07,-0.60, -0.11]

        elif 'no' in gesture_key:
            # NO: Index and middle extended upward, rest curled (index wagging side-to-side in ASL)
            base_hand[2]  = [0.12, -0.18, -0.04]; base_hand[3]  = [0.08, -0.20, -0.04]; base_hand[4]  = [0.04, -0.20, -0.04]
            base_hand[6]  = [0.08, -0.42, -0.07]; base_hand[7]  = [0.08, -0.56, -0.09]; base_hand[8]  = [0.08, -0.68, -0.11]
            base_hand[10] = [0.00, -0.44, -0.07]; base_hand[11] = [0.00, -0.58, -0.09]; base_hand[12] = [0.00, -0.70, -0.11]
            base_hand[14] = [-0.07,-0.22, -0.03]; base_hand[15] = [-0.07,-0.18, -0.02]; base_hand[16] = [-0.07,-0.15, -0.01]
            base_hand[18] = [-0.13,-0.18, -0.03]; base_hand[19] = [-0.13,-0.15, -0.02]; base_hand[20] = [-0.13,-0.12, -0.01]

        else:
            # Fallback relaxed open hand
            base_hand[2]  = [0.15, -0.22, -0.06]; base_hand[3]  = [0.22, -0.28, -0.08]; base_hand[4]  = [0.27, -0.32, -0.09]
            base_hand[6]  = [0.08, -0.42, -0.07]; base_hand[7]  = [0.08, -0.54, -0.09]; base_hand[8]  = [0.08, -0.64, -0.10]
            base_hand[10] = [0.00, -0.46, -0.07]; base_hand[11] = [0.00, -0.58, -0.09]; base_hand[12] = [0.00, -0.68, -0.10]
            base_hand[14] = [-0.07,-0.32, -0.05]; base_hand[15] = [-0.08,-0.42, -0.06]; base_hand[16] = [-0.08,-0.48, -0.07]
            base_hand[18] = [-0.13,-0.28, -0.05]; base_hand[19] = [-0.14,-0.36, -0.06]; base_hand[20] = [-0.14,-0.42, -0.07]

        # ── DATA AUGMENTATION ─────────────────────────────────────────────────
        samples = []
        for _ in range(num_samples):
            sample_hand = base_hand.copy()

            # Random 3D rotations (±20°)
            angles = np.radians(np.random.uniform(-20, 20, size=3))
            cx, cy, cz = np.cos(angles)
            sx, sy, sz = np.sin(angles)
            Rx = np.array([[1, 0, 0],  [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0,  1,   0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0],[sz, cz,  0], [0,    0, 1]])
            R  = Rz @ Ry @ Rx
            sample_hand = sample_hand @ R.T

            # Random scale (±15%)
            sample_hand *= np.random.uniform(0.85, 1.15)

            flat = sample_hand.flatten()

            # Joint-wise Gaussian jitter
            flat += np.random.normal(0, 0.03, size=63)

            # Re-zero wrist
            flat[0:3] = 0.0

            # Normalise to [-1, 1]
            max_val = np.max(np.abs(flat))
            if max_val > 0:
                flat /= max_val

            samples.append(flat)

        return np.array(samples)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING (used by Flask API when gesture list changes)
    # ─────────────────────────────────────────────────────────────────────────
    def train_model(self, db_gestures):
        """
        Trains and saves a Keras model.
        Priority order:
          1. Real collected data from ai_engine/dataset/<name>.npy
          2. Synthetic data for any class without a real .npy file (fallback).
        """
        if not db_gestures:
            print("No gestures found in database. Cannot train model.")
            return False

        self.classes = sorted([g['name'] for g in db_gestures])
        num_classes  = len(self.classes)
        print(f"Training classifier on {num_classes} classes: {self.classes}")

        x_data, y_data = [], []

        for idx, gesture_name in enumerate(self.classes):
            # ── Try real data first ───────────────────────────────────────
            dataset_file = os.path.join(DATASET_DIR, f"{gesture_name}.npy")
            if os.path.exists(dataset_file):
                real_data = np.load(dataset_file).astype(np.float32)
                if real_data.ndim == 2 and real_data.shape[1] == 63 and len(real_data) >= 10:
                    print(f"  ✓ Real data  for '{gesture_name}': {len(real_data)} samples")
                    x_data.append(real_data)
                    y_data.append(np.full(len(real_data), idx, dtype=np.int32))
                    continue

            # ── Fallback: synthetic data ──────────────────────────────────
            print(f"  ~ Synthetic for '{gesture_name}' (no real data found)")
            synth = self._generate_synthetic_landmarks(gesture_name, num_samples=1000)
            x_data.append(synth)
            y_data.append(np.full(1000, idx, dtype=np.int32))

        X = np.concatenate(x_data, axis=0)
        y = np.concatenate(y_data, axis=0)

        # Shuffle
        idx_shuffle = np.random.permutation(len(X))
        X = X[idx_shuffle]
        y = y[idx_shuffle]

        # ── Model ─────────────────────────────────────────────────────────
        model = models.Sequential([
            layers.Input(shape=(63,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax'),
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        from tensorflow.keras.callbacks import EarlyStopping
        es = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=0)

        model.fit(
            X, y,
            epochs=150,
            batch_size=32,
            validation_split=0.15,
            callbacks=[es],
            verbose=0,
        )

        self.model = model
        self.model.save(MODEL_PATH)

        classes_path = os.path.join(MODEL_DIR, 'classes.txt')
        with open(classes_path, 'w') as f:
            f.write('\n'.join(self.classes))

        print(f"Model saved to {MODEL_PATH}")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────────────────────────────────────────
    def load_model(self):
        """Loads the model and class metadata from files if they exist."""
        classes_path = os.path.join(MODEL_DIR, 'classes.txt')

        if os.path.exists(MODEL_PATH) and os.path.exists(classes_path):
            try:
                self.model = models.load_model(MODEL_PATH)
                with open(classes_path, 'r') as f:
                    self.classes = [line.strip() for line in f.read().split('\n') if line.strip()]
                print(f"Loaded classifier model with {len(self.classes)} classes: {self.classes}")
                return True
            except Exception as e:
                print(f"Failed to load saved model: {e}")

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICTION (single frame — used by Flask API)
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, normalized_landmarks):
        """
        Predicts the gesture from a single 63-float landmark vector.

        Pipeline:
          1. Run model inference.
          2. If confidence < CONFIDENCE_THRESHOLD → discard (no vote added).
          3. Append confident (class_idx, confidence) to the sliding window.
          4. Apply majority voting over the last PREDICTION_BUFFER_SIZE frames.
          5. Return winner if it has >= MAJORITY_VOTE_MIN votes, else 'Uncertain'.

        Args:
            normalized_landmarks: flat list or array of 63 floats.

        Returns:
            Tuple[str, float]:
                - class_name  : predicted gesture name, or 'Uncertain' / 'Model Not Loaded'
                - confidence  : raw model softmax confidence for the top class (0.0 – 1.0)
        """
        if self.model is None or not self.classes:
            return "Model Not Loaded", 0.0

        input_data = np.array([normalized_landmarks], dtype=np.float32)
        prediction = self.model.predict(input_data, verbose=0)[0]

        idx        = int(np.argmax(prediction))
        confidence = float(prediction[idx])

        # ── Step 1: Confidence gate ───────────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            # Low-confidence frame — do NOT add to buffer; return Uncertain
            return "Uncertain", confidence

        # ── Step 2: Add to sliding window ─────────────────────────────────
        self._pred_buffer.append(idx)

        # ── Step 3: Majority voting ────────────────────────────────────────
        if len(self._pred_buffer) < MAJORITY_VOTE_MIN:
            # Not enough confident frames yet
            return "Uncertain", confidence

        vote_counts   = Counter(self._pred_buffer)
        winner_idx, winner_votes = vote_counts.most_common(1)[0]

        if winner_votes >= MAJORITY_VOTE_MIN:
            # Compute the mean confidence for the winning class from the raw prediction
            winner_confidence = float(prediction[winner_idx])
            return self.classes[winner_idx], winner_confidence

        return "Uncertain", confidence

    # ─────────────────────────────────────────────────────────────────────────
    # BUFFER RESET (optional — call when starting a new session)
    # ─────────────────────────────────────────────────────────────────────────
    def reset_buffer(self):
        """Clears the prediction sliding-window buffer."""
        self._pred_buffer.clear()
