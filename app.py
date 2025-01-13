from flask import Flask, request, send_file, Response, send_from_directory, jsonify
from io import BytesIO
import os
import cv2
import warnings
from ultralytics import YOLO  # Import the YOLO class from ultralytics

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Flask app setup
app = Flask(__name__)

# Temporary folder for uploaded files
os.makedirs("temp", exist_ok=True)

# Load YOLOv8 model
print("Loading YOLOv8 model...")
model = YOLO('yolov8s.pt')  # Load the YOLOv8 small model
print("Model loaded successfully!")

# Function to detect objects in an image
def detect_objects_in_image(file_path):
    img = cv2.imread(file_path)

    # Perform detection
    results = model.predict(source=img, save=False, save_txt=False)

    # Draw bounding boxes and labels on the image
    for box in results[0].boxes:  # Loop through detected boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box coordinates
        conf = box.conf[0].item()  # Extract confidence score
        class_id = int(box.cls[0].item())  # Extract class ID
        label = f"{model.names[class_id]}: {conf:.2f}"  # Get label

        # Draw the bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode image as JPEG for response
    _, buffer = cv2.imencode('.jpg', img)
    return BytesIO(buffer)

# Function to generate frames for video detection
def generate_frames(video_source):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform object detection on each frame
        results = model.predict(source=frame, save=False, save_txt=False)

        # Draw bounding boxes and labels on the frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = f"{model.names[class_id]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for the main page
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def serve_css():
    return send_from_directory(os.getcwd(), 'styles.css')

# Route for object detection in images
@app.route('/detect_image', methods=['POST'])
def detect_image():
    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Perform detection and get processed image
    processed_image = detect_objects_in_image(file_path)

    # Clean up uploaded file
    os.remove(file_path)

    return send_file(processed_image, mimetype='image/jpeg')

# Route for video file detection
@app.route('/detect_video', methods=['POST'])
def detect_video():
    file = request.files['file']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Stream the video with detection
    return Response(generate_frames(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for live webcam detection
@app.route('/live_detection')
def live_detection():
    # Stream the webcam with detection
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to fetch model accuracy
@app.route('/model_accuracy', methods=['GET'])
def model_accuracy():
    # Retrieve model accuracy (mAP@0.5 and mAP@0.5:0.95)
    metrics = model.val()
    accuracy_data = {
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map
    }
    return jsonify(accuracy_data)

# Route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_name": "YOLOv8s"
    })

if __name__ == "__main__":
    app.run(debug=True)