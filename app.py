from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model_path = "C:/Users/Ben Yoon/Desktop/PythonProjects/Vision/HandPose/runs/pose/train/weights/best.pt"
model = YOLO(model_path)
password_gesture_file = 'persistence/password_gesture.npy'


def extract_hand_keypoints(results, frame_shape):
    """
    Extract hand keypoints and normalize them relative to the hand bounding box and size.
    """
    keypoints = []
    for result in results:
        if result.keypoints and len(result.boxes) > 0:
            # Extract bounding box
            bbox = result.boxes.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1

            # Extract hand keypoints
            first_hand_keypoints = result.keypoints.xy[0].cpu().numpy()
            normalized_keypoints = []

            for (x, y) in first_hand_keypoints:
                # Normalize keypoints relative to the bounding box dimensions
                norm_x = (x - x1) / width
                norm_y = (y - y1) / height
                normalized_keypoints.extend([norm_x, norm_y])

            keypoints.append(np.array(normalized_keypoints))
            break  # Only process the first detected hand

    return np.mean(keypoints, axis=0) if keypoints else np.array([])




def save_password_gesture(gesture):
    np.save(password_gesture_file, gesture)
    print("✅ Password gesture saved!")


def load_saved_gesture():
    try:
        return np.load(password_gesture_file)
    except FileNotFoundError:
        return None


def verify_password_gesture(current_gesture):
    """
    Verify the current gesture against the saved gesture.
    """
    saved_gesture = load_saved_gesture()
    if saved_gesture is None:
        print("⚠️ No saved gesture found.")
        return False

    # Compare gestures using normalized keypoints with a small tolerance
    if np.allclose(saved_gesture, current_gesture, atol=0.1):  # Adjust tolerance if needed
        print("🔓 Password Matched! Access Granted.")
        return True
    else:
        print("❌ Password Mismatch! Try again.")
        return False




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture-gesture', methods=['POST'])
def capture_gesture():
    try:
        image_data = request.data
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'Failed to decode image'})

        results = model(frame)
        keypoints = extract_hand_keypoints(results, frame.shape)

        if keypoints.size > 0:
            save_password_gesture(keypoints)
            return jsonify({'status': 'Gesture Saved'})
        return jsonify({'status': 'No valid gesture detected'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'Server error', 'error': str(e)}), 500


@app.route('/verify-gesture', methods=['POST'])
def verify_gesture():
    try:
        image_data = request.data
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'Failed to decode image'})

        results = model(frame)
        keypoints = extract_hand_keypoints(results, frame.shape)

        if keypoints.size > 0 and verify_password_gesture(keypoints):
            return jsonify({'status': 'Access Granted'})
        return jsonify({'status': 'Access Denied'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'Server error', 'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
