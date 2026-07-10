import os
import sys
import base64
import cv2
import numpy as np
import pyttsx3
import threading
from flask import Blueprint, request, jsonify

# Add project root directory to path to resolve imports correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai_engine.hand_detection import HandDetector
from ai_engine.gesture_model import GestureClassifier
from backend.database import db_manager

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Initialize AI instances
detector = HandDetector()
classifier = GestureClassifier()

# Thread lock for pyttsx3 speech engine (prevents concurrent access issues)
tts_lock = threading.Lock()

def speak_synchronously(text):
    """Speaks the given text using pyttsx3 in a thread-safe, COM-initialized manner, blocking until complete."""
    import pythoncom
    pythoncom.CoInitialize()
    try:
        with tts_lock:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            return True
    except Exception as e:
        print(f"Server-side TTS error: {e}")
        return False
    finally:
        pythoncom.CoUninitialize()

def initialize_ai():
    """Initializes the AI model, training it if not already saved or if classes differ."""
    db_manager.init_db()
    model_loaded = classifier.load_model()
    gestures = db_manager.get_all_gestures()
    db_classes = sorted([g['name'] for g in gestures])
    
    if not model_loaded or sorted(classifier.classes) != db_classes:
        print("Model missing or class mismatch. Retraining classifier model on database gestures...")
        classifier.train_model(gestures)

# Run initialization
initialize_ai()

@api_bp.route('/status', methods=['GET'])
def get_status():
    """Returns the initialization and state status of backend resources."""
    model_loaded = classifier.model is not None
    gestures = db_manager.get_all_gestures()
    
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "classes_count": len(classifier.classes),
        "classes": classifier.classes,
        "database_connected": True,
        "total_gestures_in_db": len(gestures)
    })


@api_bp.route('/predict', methods=['POST'])
def predict_gesture():
    """
    Accepts a base64 encoded video frame, detects hands, extracts landmarks,
    classifies the gesture, and returns the classification results.
    """
    try:
        print("========== NEW PREDICTION REQUEST ==========")

        data = request.get_json()

        if not data:
            print("ERROR: No JSON received")
            return jsonify({"error": "No JSON received"}), 400

        if 'image' not in data:
            print("ERROR: No image key in request")
            return jsonify({"error": "No image data provided"}), 400

        print("Received image successfully")

        # Decode base64 image
        image_data = data['image']

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("ERROR: Frame decoding failed")
            return jsonify({"error": "Failed to decode frame"}), 400

        print("Frame decoded successfully")
        print("Frame shape:", frame.shape)

        # Detect hand
        normalized_landmarks, raw_landmarks = detector.process_frame(frame)

        if normalized_landmarks is None:
            print("No hand detected")

            return jsonify({
                "detected": False,
                "gesture": "No hand detected",
                "confidence": 0.0,
                "landmarks": None
            })

        print("Hand detected")
        print("Landmark count:", len(normalized_landmarks))

        # Predict
        print("Calling classifier.predict()...")

        gesture_name, confidence = classifier.predict(normalized_landmarks)

        print("Prediction completed")
        print("Gesture:", gesture_name)
        print("Confidence:", confidence)

        history_id = None

        if gesture_name not in ["Uncertain", "Model Not Loaded"]:
            print("Saving prediction to database...")
            history_id = db_manager.add_history_entry(
                gesture_name,
                confidence,
                spoken=0
            )

        print("Returning response")

        return jsonify({
            "detected": True,
            "gesture": gesture_name,
            "confidence": confidence,
            "landmarks": raw_landmarks,
            "history_id": history_id
        })

    except Exception as e:
        import traceback

        print("========== PREDICTION ERROR ==========")
        print(str(e))
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

@api_bp.route('/gestures', methods=['GET', 'POST'])
def manage_gestures():
    """Retrieves or adds a gesture to the library."""
    if request.method == 'GET':
        gestures = db_manager.get_all_gestures()
        return jsonify(gestures)
        
    elif request.method == 'POST':
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"error": "Missing gesture name"}), 400
            
        name = data['name'].strip()
        description = data.get('description', '').strip()
        
        if not name:
            return jsonify({"error": "Gesture name cannot be empty"}), 400
            
        success = db_manager.add_gesture(name, description)
        if not success:
            return jsonify({"error": "Gesture already exists"}), 400
            
        # Retrain the TensorFlow Keras classifier with the updated list of gestures
        gestures = db_manager.get_all_gestures()
        retrain_success = classifier.train_model(gestures)
        
        return jsonify({
            "success": True,
            "message": f"Gesture '{name}' added successfully.",
            "model_retrained": retrain_success
        })

@api_bp.route('/gestures/<string:name>', methods=['DELETE'])
def delete_gesture(name):
    """Deletes a gesture from the library and triggers classifier retraining."""
    success = db_manager.delete_gesture(name)
    if not success:
        return jsonify({"error": "Gesture not found"}), 404
        
    # Retrain TensorFlow model with the remaining list of gestures
    gestures = db_manager.get_all_gestures()
    if gestures:
        classifier.train_model(gestures)
    else:
        # If no gestures left, delete the model file to avoid load issues
        model_path = classifier.MODEL_PATH if hasattr(classifier, 'MODEL_PATH') else os.path.join(PROJECT_ROOT, 'ai_engine', 'model', 'gesture_classifier.keras')
        if os.path.exists(model_path):
            os.remove(model_path)
        classifier.model = None
        classifier.classes = []
        
    return jsonify({
        "success": True,
        "message": f"Gesture '{name}' deleted and model retrained."
    })

@api_bp.route('/history', methods=['GET', 'DELETE'])
def manage_history():
    """Retrieves or clears the recognition history log."""
    if request.method == 'GET':
        history = db_manager.get_history()
        return jsonify(history)
        
    elif request.method == 'DELETE':
        db_manager.clear_history()
        return jsonify({"success": True, "message": "History cleared."})

@api_bp.route('/speak', methods=['POST'])
def speak():
    """Triggers server-side text-to-speech output using pyttsx3 synchronously."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing speech text"}), 400
        
    text = data['text'].strip()
    history_id = data.get('history_id')
    
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
        
    # Speak synchronously and check success
    success = speak_synchronously(text)
    
    if not success:
        return jsonify({"error": "Failed to output voice on server"}), 500
        
    # Update SQLite database log if history_id is provided
    if history_id is not None:
        db_manager.update_spoken_status(history_id, spoken=1)
        
    return jsonify({"success": True, "message": f"Speech completed: '{text}'"})
