import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """Initializes the MediaPipe Hands object."""

        # Debug information for Render logs
        print("MediaPipe version:", getattr(mp, "__version__", "Unknown"))
        print("MediaPipe file:", getattr(mp, "__file__", "Unknown"))
        print("Has solutions:", hasattr(mp, "solutions"))

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame_bgr):
        """
        Processes a BGR image frame and extracts hand landmarks.
        Returns:
            - normalized_flat_landmarks: list of 63 floats (relative and scaled), or None if no hand detected.
            - raw_landmarks: list of 21 dicts with x, y, z coordinates in image ratio [0, 1], or None if no hand detected.
        """
        # Convert BGR image to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract raw landmarks
        raw_landmarks = []
        for lm in hand_landmarks.landmark:
            raw_landmarks.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z
            })

        # Normalize landmarks
        wrist = hand_landmarks.landmark[0]
        temp_coords = []

        for lm in hand_landmarks.landmark:
            temp_coords.append(lm.x - wrist.x)
            temp_coords.append(lm.y - wrist.y)
            temp_coords.append(lm.z - wrist.z)

        max_val = max(abs(val) for val in temp_coords)
        if max_val == 0:
            max_val = 1.0

        normalized_flat_landmarks = [val / max_val for val in temp_coords]

        return normalized_flat_landmarks, raw_landmarks