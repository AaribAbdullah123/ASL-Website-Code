import os
import urllib.request
import webbrowser
from flask import Flask, jsonify, render_template, request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn

os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("asl_dataset", exist_ok=True)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASL Neural Translator</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; overflow-y: auto; padding: 20px; }
        .container { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.1); padding: 30px; border-radius: 20px; width: 480px; text-align: center; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }
        h1 { font-size: 24px; margin-bottom: 6px; font-weight: 600; color: #38bdf8; }
        p { font-size: 13px; color: #94a3b8; margin-bottom: 20px; }
        .tab-bar { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; }
        .tab-btn { background: rgba(15, 23, 42, 0.6); border: 1px solid #475569; color: #cbd5e1; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; transition: all 0.2s; }
        .tab-btn.active { background: #38bdf8; color: #0f172a; border-color: #38bdf8; font-weight: 600; }
        .mode-section { display: none; }
        .mode-section.active { display: block; }
        .drop-zone { border: 2px dashed #475569; border-radius: 12px; padding: 30px 20px; cursor: pointer; transition: all 0.3s ease; background: rgba(15, 23, 42, 0.4); }
        .drop-zone.dragover { border-color: #38bdf8; background: rgba(56, 189, 248, 0.05); }
        .drop-zone p { margin: 0; color: #cbd5e1; }
        input[type="file"] { display: none; }
        .webcam-box { position: relative; width: 100%; border-radius: 12px; overflow: hidden; background: #000; margin-bottom: 15px; border: 1px solid #475569; }
        video { width: 100%; display: block; transform: scaleX(-1); }
        .controls { display: flex; gap: 10px; justify-content: center; }
        .btn { padding: 10px 20px; border-radius: 8px; border: none; font-weight: 600; cursor: pointer; font-size: 14px; transition: opacity 0.2s; }
        .btn-start { background: #22c55e; color: #fff; }
        .btn-stop { background: #ef4444; color: #fff; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-box { margin-top: 25px; padding: 15px; background: rgba(15, 23, 42, 0.8); border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.2); display: none; }
        .result-box h2 { font-size: 28px; color: #4ade80; margin-bottom: 4px; }
        .result-box span { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .loader { border: 3px solid rgba(255,255,255,0.1); border-top: 3px solid #38bdf8; border-radius: 50%; width: 28px; height: 28px; animation: spin 1s linear infinite; margin: 15px auto 0; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>ASL Neural Translator</h1>
        <p>Offline BiLSTM + MediaPipe Vision Engine</p>
        <div class="tab-bar">
            <button class="tab-btn active" onclick="switchMode('live')">Live Record</button>
            <button class="tab-btn" onclick="switchMode('upload')">Upload Video</button>
        </div>
        <div id="liveMode" class="mode-section active">
            <div class="webcam-box">
                <video id="webcam" autoplay muted playsinline></video>
            </div>
            <div class="controls">
                <button id="startBtn" class="btn btn-start" onclick="startRecording()">Start Recording</button>
                <button id="stopBtn" class="btn btn-stop" onclick="stopRecording()" disabled>Stop & Translate</button>
            </div>
        </div>
        <div id="uploadMode" class="mode-section">
            <div class="drop-zone" id="dropZone">
                <p>Drag & Drop video file here or <strong>Browse</strong></p>
                <input type="file" id="fileInput" accept="video/*">
            </div>
        </div>
        <div class="loader" id="loader"></div>
        <div class="result-box" id="resultBox">
            <span>Translated Sign</span>
            <h2 id="signOutput">--</h2>
            <span id="confOutput">Confidence: 0.0%</span>
        </div>
    </div>
    <script>
        let mediaRecorder;
        let recordedChunks = [];
        let webcamStream = null;
        const webcamElement = document.getElementById('webcam');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const loader = document.getElementById('loader');
        const resultBox = document.getElementById('resultBox');
        const signOutput = document.getElementById('signOutput');
        const confOutput = document.getElementById('confOutput');

        async function initWebcam() {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
                webcamElement.srcObject = webcamStream;
            } catch (err) { console.error("Webcam access denied:", err); }
        }
        initWebcam();

        function switchMode(mode) {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.mode-section').forEach(s => s.classList.remove('active'));
            if(mode === 'live') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('liveMode').classList.add('active');
            } else {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('uploadMode').classList.add('active');
            }
        }

        function startRecording() {
            recordedChunks = [];
            try { mediaRecorder = new MediaRecorder(webcamStream, { mimeType: 'video/webm;codecs=vp9' }); }
            catch (e) { mediaRecorder = new MediaRecorder(webcamStream); }
            mediaRecorder.ondataavailable = (event) => { if (event.data.size > 0) recordedChunks.push(event.data); };
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'video/webm' });
                sendToServer(new File([blob], "sign_recording.webm", { type: 'video/webm' }));
            };
            mediaRecorder.start();
            startBtn.disabled = true;
            stopBtn.disabled = false;
            resultBox.style.display = 'none';
        }

        function stopRecording() {
            mediaRecorder.stop();
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) sendToServer(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) sendToServer(fileInput.files[0]); });

        function sendToServer(file) {
            const formData = new FormData();
            formData.append('file', file);
            loader.style.display = 'block';
            resultBox.style.display = 'none';

            fetch('/predict', { method: 'POST', body: formData })
            .then(res => res.json())
            .then(data => {
                loader.style.display = 'none';
                if(data.error) { alert(data.error); return; }
                signOutput.textContent = data.sign;
                confOutput.textContent = `Confidence: ${data.confidence}%`;
                resultBox.style.display = 'block';
            })
            .catch(err => {
                loader.style.display = 'none';
                alert('Inference pipeline error occurred.');
            });
        }
    </script>
</body>
</html>
"""

html_path = os.path.join("templates", "index.html")
if not os.path.exists(html_path):
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

model_task_path = "hand_landmarker.task"
if not os.path.exists(model_task_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_task_path
    )

base_options = python.BaseOptions(model_asset_path=model_task_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.75
)
detector = vision.HandLandmarker.create_from_options(options)

actions = [
    "hello", "thank", "please", "yes", "no", "help", "good", "bad", "more", "stop",
    "sorry", "name", "want", "like", "love", "eat", "drink", "water", "food", "friend",
    "family", "home", "work", "school", "time", "day", "night", "today", "now", "where",
    "who", "what", "when", "why", "how", "fast", "slow", "big", "small", "hot",
    "cold", "happy", "sad", "open", "close", "read", "write", "learn", "computer", "understand"
]
sequence_length = 90

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[INFO] Device: {device}")

class ASLBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(ASLBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = ASLBiLSTM(input_dim=63, hidden_dim=128, output_dim=len(actions)).to(device)
if os.path.exists("asl_bilstm_model.pth"):
    model.load_state_dict(torch.load("asl_bilstm_model.pth", map_location=device))
model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    cap = cv2.VideoCapture(file_path)
    window_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
            wrist = landmarks[0:3]
            normalized = landmarks - np.tile(wrist, 21)
            window_data.append(normalized)
        else:
            window_data.append(np.zeros(63))

    cap.release()

    if os.path.exists(file_path):
        os.remove(file_path)

    if len(window_data) == 0:
        return jsonify({"error": "Could not extract frames"}), 400

    if len(window_data) < sequence_length:
        while len(window_data) < sequence_length:
            window_data.append(np.zeros(63))
    else:
        window_data = window_data[:sequence_length]

    input_tensor = torch.tensor(np.array(window_data), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_sign = actions[predicted_idx.item()]
        conf_score = round(confidence.item() * 100, 2)

    return jsonify({"sign": predicted_sign.upper(), "confidence": conf_score})

if __name__ == "__main__":
    print("[INFO] Starting Flask server...")
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
