import os
import sys
import base64
import traceback
import cv2
import numpy as np
import pyttsx3
import threading

from flask import Blueprint, request, jsonify

# ------------------------------------------------------------------
# Project Path
# ------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

from ai_engine.hand_detection import HandDetector
from ai_engine.gesture_model import GestureClassifier
from backend.database import db_manager

# ------------------------------------------------------------------
# Blueprint
# ------------------------------------------------------------------

api_bp = Blueprint("api", __name__)

# ------------------------------------------------------------------
# AI
# ------------------------------------------------------------------

detector = HandDetector()
classifier = GestureClassifier()

# ------------------------------------------------------------------
# TTS
# ------------------------------------------------------------------

tts_lock = threading.Lock()


def speak_synchronously(text):
    import pythoncom

    pythoncom.CoInitialize()

    try:
        with tts_lock:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            return True

    except Exception as e:
        print("TTS Error:", e)
        return False

    finally:
        pythoncom.CoUninitialize()


# ------------------------------------------------------------------
# Initialize Model
# ------------------------------------------------------------------

def initialize_ai():

    db_manager.init_db()

    model_loaded = classifier.load_model()

    gestures = db_manager.get_all_gestures()

    db_classes = sorted([g["name"] for g in gestures])

    if (not model_loaded) or (sorted(classifier.classes) != db_classes):

        print("Retraining model...")

        classifier.train_model(gestures)


initialize_ai()


# ------------------------------------------------------------------
# Status
# ------------------------------------------------------------------

@api_bp.route("/status", methods=["GET"])
def get_status():

    gestures = db_manager.get_all_gestures()

    return jsonify({
        "status": "online",
        "model_loaded": classifier.model is not None,
        "classes_count": len(classifier.classes),
        "classes": classifier.classes,
        "database_connected": True,
        "total_gestures_in_db": len(gestures)
    })


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

@api_bp.route("/predict", methods=["POST"])
def predict_gesture():

    try:

        print("\n==============================")
        print("NEW PREDICTION REQUEST")
        print("==============================")

        data = request.get_json()

        if data is None:

            print("No JSON received")

            return jsonify({
                "error": "No JSON received"
            }), 400

        if "image" not in data:

            print("Image key missing")

            return jsonify({
                "error": "No image supplied"
            }), 400

        image_data = data["image"]

        print("Image received")

        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)

        np_arr = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:

            print("Frame decode failed")

            return jsonify({
                "error": "Image decode failed"
            }), 400

        print("Frame shape:", frame.shape)

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

        print("Landmarks:", len(normalized_landmarks))

        print("Calling classifier...")

        gesture_name, confidence = classifier.predict(normalized_landmarks)

        print("Prediction Finished")

        print("Gesture:", gesture_name)

        print("Confidence:", confidence)

        history_id = None

        if gesture_name not in [
            "Uncertain",
            "Model Not Loaded"
        ]:

            history_id = db_manager.add_history_entry(
                gesture_name,
                confidence,
                spoken=0
            )

        print("Returning JSON")

        return jsonify({

            "detected": True,

            "gesture": gesture_name,

            "confidence": confidence,

            "landmarks": raw_landmarks,

            "history_id": history_id

        })

    except Exception as e:

        print("\n========== ERROR ==========")

        traceback.print_exc()

        return jsonify({

            "error": str(e)

        }), 500
    # ------------------------------------------------------------------
# Gesture Management
# ------------------------------------------------------------------

@api_bp.route("/gestures", methods=["GET", "POST"])
def manage_gestures():

    if request.method == "GET":

        gestures = db_manager.get_all_gestures()

        return jsonify(gestures)

    # ---------------- POST ---------------- #

    data = request.get_json()

    if not data:

        return jsonify({
            "error": "No JSON received"
        }), 400

    if "name" not in data:

        return jsonify({
            "error": "Gesture name missing"
        }), 400

    name = data["name"].strip()

    description = data.get("description", "").strip()

    if name == "":

        return jsonify({
            "error": "Gesture name cannot be empty"
        }), 400

    success = db_manager.add_gesture(
        name,
        description
    )

    if not success:

        return jsonify({
            "error": "Gesture already exists"
        }), 400

    print("Retraining after adding gesture...")

    gestures = db_manager.get_all_gestures()

    retrained = classifier.train_model(gestures)

    return jsonify({

        "success": True,

        "message": f"{name} added successfully.",

        "model_retrained": retrained

    })


# ------------------------------------------------------------------
# Delete Gesture
# ------------------------------------------------------------------

@api_bp.route("/gestures/<string:name>", methods=["DELETE"])
def delete_gesture(name):

    success = db_manager.delete_gesture(name)

    if not success:

        return jsonify({

            "error": "Gesture not found"

        }), 404

    gestures = db_manager.get_all_gestures()

    if len(gestures) > 0:

        print("Retraining after deleting gesture...")

        classifier.train_model(gestures)

    else:

        classifier.model = None

        classifier.classes = []

    return jsonify({

        "success": True,

        "message": f"{name} deleted successfully."

    })


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------

@api_bp.route("/history", methods=["GET", "DELETE"])
def manage_history():

    if request.method == "GET":

        history = db_manager.get_history()

        return jsonify(history)

    db_manager.clear_history()

    return jsonify({

        "success": True,

        "message": "History cleared."

    })
# ------------------------------------------------------------------
# Text To Speech
# ------------------------------------------------------------------

@api_bp.route("/speak", methods=["POST"])
def speak():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No JSON received"
            }), 400

        text = data.get("text", "").strip()

        history_id = data.get("history_id")

        if text == "":

            return jsonify({
                "error": "Text cannot be empty"
            }), 400

        print("Speaking:", text)

        success = speak_synchronously(text)

        if not success:

            return jsonify({
                "error": "Speech failed"
            }), 500

        if history_id is not None:

            db_manager.update_spoken_status(
                history_id,
                spoken=1
            )

        return jsonify({

            "success": True,

            "message": f"Speech completed: {text}"

        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error": str(e)

        }), 500


# ------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------

@api_bp.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "Sign Speak AI Backend Running",

        "model_loaded": classifier.model is not None,

        "total_classes": len(classifier.classes)

    })
# ------------------------------------------------------------------
# Error Handlers
# ------------------------------------------------------------------

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "API endpoint not found"
    }), 404


@api_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405


@api_bp.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


print("=" * 60)
print("Sign Speak AI API Loaded Successfully")
print("Available Routes:")
print("  GET    /")
print("  GET    /status")
print("  POST   /predict")
print("  GET    /gestures")
print("  POST   /gestures")
print("  DELETE /gestures/<name>")
print("  GET    /history")
print("  DELETE /history")
print("  POST   /speak")
print("=" * 60)