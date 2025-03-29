import os
import absl.logging
import tensorflow as tf

# Disable all TensorFlow logging except errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize ABSL logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.use_absl_handler()

from flask import Flask, render_template, Response, jsonify, request, redirect
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr
from PIL import Image
import random
import base64

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize Flask app
app = Flask(__name__)

# Define classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

# Configure model loading with explicit signatures
@tf.function(experimental_relax_shapes=True)
def load_model_with_signatures(model_path):
    return load_model(model_path)

# MediaPipe settings for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the trained model for hand sign recognition
model_path = os.path.join("hand sign model cnn tensorflow", "hand_landmarks.h5")
print(f"Looking for model at: {os.path.abspath(model_path)}")

if not os.path.exists(model_path):
    print(f"WARNING: Model file not found at {model_path}")
    print("The application will continue but hand sign recognition will not work.")
    model = None
else:
    try:
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

recognizer = sr.Recognizer()

word_to_number = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "zero": 0,
    "ten": 10
}

# Open the camera
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert the frame from BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands in the frame
            results = hands.process(frame_rgb)

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Encode the frame as a JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Return the image as a stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/voice-learning')
def voiceLearning():
    return render_template('learningByAudio.html')

@app.route('/learning-letters')
def learningletters():
    return render_template('learningLetter.html')

@app.route('/test')
def test():
    return render_template('selfTest.html')

@app.route('/practicing')
def practicing():
    return render_template('practice.html')

@app.route('/learningName')
def learningName():
    return render_template('learningName.html')

@app.route('/cardGame')
def cardGame():
    return render_template('cardGame.html')

@app.route('/learning')
def learning():
    return render_template('learning.html')

@app.route('/ai-converse')
def ai_converse():
    # Redirect to the desired local server or directory
    return redirect('http://localhost:3000')

@app.route('/video_feed')
def video_feed():
    # Return the video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the file path.'}), 500

    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Could not capture image from camera'}), 500
    else:
        # Convert the frame from BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect hand landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                # Convert landmarks to a NumPy array and add a new dimension
                input_data = np.array(landmarks).reshape(1, 21, 3)

                # Predict the class using the model
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)[0]

                # Display the corresponding character based on the prediction
                predicted_character = classes[predicted_class]
            return (f'{predicted_character}')
        else:
            return jsonify({'error': 'No hand detected in frame'}), 400

@app.route('/speech_recognition', methods=['GET'])
def speech_recognition():
    try:
        while True:
            with sr.Microphone() as source:
                print("Say something...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio, language="en-US")
            print(f"The computer heard: {text}")
            if text.lower().startswith("letter "):
                character = text.split()[1].lower()
                image_folder = os.path.join("static", "images", "hand sign none")
                image_path = os.path.join(image_folder, f"{character}.png")
                print(image_path)
                if os.path.exists(image_path):
                    return jsonify({ "image": f"{character}.png"})
                else:
                    return jsonify({"message": f"No image found for the letter {character}"})
            elif text.lower().startswith("number "):
                character = text.split()[1].lower()
                image_folder = os.path.join("static", "images", "hand sign none")
                image_path = os.path.join(image_folder, f"{character}.png")
                print(image_path)
                if os.path.exists(image_path):
                    return jsonify({ "image": f"{character}.png"})
                else:
                    return jsonify({"message": f"No image found for the number {character}"})
            else:
                print("Could not identify a letter or number")
                return jsonify({"message": "Could not identify a letter or number"})
    except sr.UnknownValueError:
        return jsonify({"message": "Could not understand what you said"})
    except sr.RequestError as e:
        return jsonify({"message": f"Error connecting to the recognition service: {e}"})

letter_to_image = {
    'a': 'static/images/Hand signs/a.png',
    'b': 'static/images/Hand signs/b.png',
    'c': 'static/images/Hand signs/c.png',
    'd': 'static/images/Hand signs/d.png',
    'e': 'static/images/Hand signs/e.png',
    'f': 'static/images/Hand signs/f.png',
    'g': 'static/images/Hand signs/g.png',
    'h': 'static/images/Hand signs/h.png',
    'i': 'static/images/Hand signs/i.png',
    'j': 'static/images/Hand signs/j.png',
    'k': 'static/images/Hand signs/k.png',
    'l': 'static/images/Hand signs/l.png',
    'm': 'static/images/Hand signs/m.png',
    'n': 'static/images/Hand signs/n.png',
    'o': 'static/images/Hand signs/o.png',
    'p': 'static/images/Hand signs/p.png',
    'q': 'static/images/Hand signs/q.png',
    'r': 'static/images/Hand signs/r.png',
    's': 'static/images/Hand signs/s.png',
    't': 'static/images/Hand signs/t.png',
    'u': 'static/images/Hand signs/u.png',
    'v': 'static/images/Hand signs/v.png',
    'w': 'static/images/Hand signs/w.png',
    'x': 'static/images/Hand signs/x.png',
    'y': 'static/images/Hand signs/y.png',
    'z': 'static/images/Hand signs/z.png'
}

# Global variables to store the user's name and image URLs
user_name = ""
image_urls = []

@app.route('/save_name', methods=['POST', 'GET'])
def save_name():
    global user_name, image_urls  # Add the array for image URLs

    # If the user clicks the 'clear' button
    if request.method == 'POST' and 'action' in request.form and request.form['action'] == 'clear':
        user_name = ""  # Clear the name
        image_urls = []  # Clear the array of image URLs
        return render_template('learningName.html', images=None)  # Display the page without images

    # If the user clicks the 'save' button
    elif request.method == 'POST' and 'action' in request.form and request.form['action'] == 'save':
        user_name = request.form['username'].lower()  # Get the name from the form and convert it to lowercase

        # Create an array of image URLs in the correct order
        image_urls = []
        for letter in user_name:
            if letter in letter_to_image:
                image_urls.append(letter_to_image[letter])  # Add the image to the array in the correct order

        return render_template('learningName.html', images=image_urls)  # Display the images in the HTML

    # In case of GET (when the page is refreshed)
    else:
        user_name = ""  # Reset the name
        image_urls = []  # Reset the images
        return render_template('learningName.html', images=None)  # Display the empty page

@app.route('/capture')
def capture():
    if model is None:
        # Return a message and continue with camera but no prediction
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
        cv2.putText(frame, "Model not loaded", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_str, 'prediction': 'Model not loaded', 'error': 'Model file not found'})

    success, frame = camera.read()
    if not success:
        return jsonify({'error': 'Failed to capture image from camera'}), 500
    else:
        # Convert the frame from BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(frame_rgb)

        predicted_character = 'No hand detected'
        
        # Draw hand landmarks on the frame even if no hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect hand landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                # Convert landmarks to a NumPy array and add a new dimension
                input_data = np.array(landmarks).reshape(1, 21, 3)

                # Predict the class using the model
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)[0]

                # Display the corresponding character based on the prediction
                predicted_character = classes[predicted_class]

                # Draw the character on the image
                cv2.putText(frame, predicted_character, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert back to BGR to display the image
        ret, buffer = cv2.imencode('.jpg', frame)

        img_str = base64.b64encode(buffer).decode('utf-8')  # Convert the image to base64 for display in HTML
        # Return the image and prediction to the client
        return jsonify({'image': img_str, 'prediction': predicted_character})

@app.route('/random_character', methods=['GET'])
def random_character_endpoint():
    global random_character
    random_character = random.choice(classes)  # Pick a new random character
    return jsonify({'random_character': random_character})

@app.route('/check_prediction', methods=['POST'])
def check_prediction():
    data = request.get_json()
    predicted_character = data.get('predicted_character')

    if predicted_character == random_character:
        result = 'correct'
    else:
        result = 'uncorrect'

    return jsonify({'result': result})

@app.route('/video_feed_pilot')
def video_feed_pilot():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        # Set default environment variables for AI Converse if not already set
        if 'AI_CONVERSE_PORT' not in os.environ:
            os.environ['AI_CONVERSE_PORT'] = '3000'
        if 'AI_CONVERSE_HOST' not in os.environ:
            os.environ['AI_CONVERSE_HOST'] = 'localhost'
        if 'AI_CONVERSE_PATH' not in os.environ:
            os.environ['AI_CONVERSE_PATH'] = '/dist/ai-converse/browser'
            
        # First try default port
        port = 5000
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"Attempting to start server on port {port}")
                print(f"AI Converse will be accessed at: http://{os.environ['AI_CONVERSE_HOST']}:{os.environ['AI_CONVERSE_PORT']}{os.environ['AI_CONVERSE_PATH']}")
                app.run(
                    host='127.0.0.1',
                    port=port,
                    debug=True,
                    use_reloader=False  # Disable reloader to prevent handle issues
                )
                break
            except OSError as e:
                print(f"Port {port} is busy, trying next port")
                port += 1
                if attempt == max_attempts - 1:
                    raise e
                
    except Exception as e:
        print(f"Failed to start server: {e}")
