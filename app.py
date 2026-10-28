import os
import io
import time
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_320 = BASE_DIR / "runs" / "yolov8n_320" / "weights" / "best.pt"
WEIGHTS_640 = BASE_DIR / "runs" / "yolov8n_640" / "weights" / "best.pt"

# Cache loaded models
models = {}

def get_model(model_type="320"):
    if model_type not in models:
        model_path = WEIGHTS_640 if model_type == "640" and WEIGHTS_640.exists() else WEIGHTS_320
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        models[model_type] = YOLO(str(model_path))
    return models[model_type]

# Color mapping for classes (RGB)
CLASS_COLORS = {
    "bent": "#ef4444",    # red
    "color": "#f59e0b",   # amber
    "flip": "#8b5cf6",    # purple
    "scratch": "#06b6d4"  # cyan
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MVTec Metal Nut Defect Detector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(22, 27, 46, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.35);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 50% 0%, #1e293b, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 20px 60px 20px;
    }
    header {
      text-align: center;
      margin-bottom: 28px;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #60a5fa, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 8px;
    }
    header p {
      color: var(--text-muted);
      font-size: 0.95rem;
    }
    .badge {
      display: inline-block;
      padding: 3px 10px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-radius: 9999px;
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      margin-bottom: 10px;
    }
    .main-container {
      width: 100%;
      max-width: 1080px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }
    @media (max-width: 820px) {
      .main-container {
        grid-template-columns: 1fr;
      }
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .card-title {
      font-size: 1.1rem;
      font-weight: 600;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    /* Upload Box */
    .upload-area {
      border: 2px dashed rgba(255, 255, 255, 0.18);
      border-radius: 12px;
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      background: rgba(15, 23, 42, 0.4);
    }
    .upload-area:hover, .upload-area.dragover {
      border-color: var(--accent);
      background: rgba(59, 130, 246, 0.08);
    }
    .upload-area svg {
      width: 44px;
      height: 44px;
      color: #60a5fa;
      margin-bottom: 12px;
    }
    .upload-area p {
      font-size: 0.95rem;
      color: var(--text-muted);
    }
    .upload-area span {
      font-size: 0.8rem;
      color: #64748b;
    }
    /* Controls */
    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    label {
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-muted);
    }
    select, input[type="range"] {
      width: 100%;
    }
    select {
      background: #1e293b;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      padding: 8px 12px;
      color: var(--text);
      font-size: 0.9rem;
      outline: none;
    }
    .slider-val {
      font-size: 0.85rem;
      color: #60a5fa;
      font-weight: 600;
    }
    .btn-run {
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn-run:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 4px 15px var(--accent-glow);
    }
    .btn-run:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    /* Sample Chips */
    .samples-wrap {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .sample-chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 0.78rem;
      cursor: pointer;
      color: #cbd5e1;
      transition: all 0.15s;
    }
    .sample-chip:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: #3b82f6;
      color: white;
    }
    /* Result Section */
    .preview-box {
      min-height: 280px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(15, 23, 42, 0.5);
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow: hidden;
      position: relative;
    }
    .preview-box img {
      max-width: 100%;
      max-height: 380px;
      object-fit: contain;
      border-radius: 8px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .status-good {
      background: rgba(34, 197, 94, 0.15);
      color: #4ade80;
      border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-defect {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .detections-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    .detections-table th, .detections-table td {
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .detections-table th {
      color: var(--text-muted);
      font-weight: 500;
    }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .loader {
      display: none;
      width: 32px;
      height: 32px;
      border: 3px solid rgba(255,255,255,0.2);
      border-top-color: #3b82f6;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <div class="badge">Edge Vision AI</div>
    <h1>MVTec Metal Nut Defect Detector</h1>
    <p>Upload or test any metal nut sample for instant defect localization and classification</p>
  </header>

  <div class="main-container">
    <!-- Left Card: Input & Controls -->
    <div class="card">
      <div class="card-title">
        <span>📸</span> Input Image & Parameters
      </div>

      <div class="upload-area" id="dropArea">
        <svg fill="none" stroke="currentColor" stroke-width="1.8" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
        <p><strong>Click to browse</strong> or drag & drop image</p>
        <span>Supports PNG, JPG, JPEG</span>
        <input type="file" id="fileInput" accept="image/*" style="display: none;">
      </div>

      <div>
        <label style="margin-bottom: 6px; display: block;">Quick Test Samples:</label>
        <div class="samples-wrap" id="samplesContainer">
          <button class="sample-chip" data-sample="good">Defect Free (Good)</button>
          <button class="sample-chip" data-sample="scratch">Scratch Sample</button>
          <button class="sample-chip" data-sample="bent">Bent Sample</button>
          <button class="sample-chip" data-sample="flip">Flip Sample</button>
          <button class="sample-chip" data-sample="color">Color Sample</button>
        </div>
      </div>

      <div class="controls">
        <div class="control-group">
          <label for="modelSelect">Model Checkpoint</label>
          <select id="modelSelect">
            <option value="320">YOLOv8n-320 (Lightweight Edge)</option>
            <option value="640">YOLOv8n-640 (High Resolution)</option>
          </select>
        </div>
        <div class="control-group">
          <label for="confSlider">
            Confidence Threshold: <span class="slider-val" id="confVal">0.25</span>
          </label>
          <input type="range" id="confSlider" min="0.10" max="0.90" step="0.05" value="0.25">
        </div>
      </div>

      <button class="btn-run" id="btnRun" disabled>
        <span>Detect Defects</span>
      </button>
    </div>

    <!-- Right Card: Output & Analysis -->
    <div class="card">
      <div class="card-title" style="justify-content: space-between;">
        <span style="display: flex; align-items: center; gap: 8px;">
          <span>🎯</span> Detection Results
        </span>
        <div id="statusBadge" style="display: none;"></div>
      </div>

      <div class="preview-box" id="previewBox">
        <div class="loader" id="loader"></div>
        <span id="previewPlaceholder" style="color: #64748b; font-size: 0.9rem;">
          Upload an image to see detection overlay
        </span>
        <img id="resultImage" style="display: none;" alt="Detection Result">
      </div>

      <div id="statsBox" style="display: none; display: flex; justify-content: space-between; font-size: 0.85rem; color: #94a3b8; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px;">
        <span id="latencyText">Latency: -</span>
        <span id="countText">Boxes: 0</span>
      </div>

      <div id="tableContainer" style="display: none; max-height: 180px; overflow-y: auto;">
        <table class="detections-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Confidence</th>
              <th>Bounding Box (x1, y1, x2, y2)</th>
            </tr>
          </thead>
          <tbody id="detectionsList"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('fileInput');
    const dropArea = document.getElementById('dropArea');
    const btnRun = document.getElementById('btnRun');
    const confSlider = document.getElementById('confSlider');
    const confVal = document.getElementById('confVal');
    const modelSelect = document.getElementById('modelSelect');
    const previewPlaceholder = document.getElementById('previewPlaceholder');
    const resultImage = document.getElementById('resultImage');
    const loader = document.getElementById('loader');
    const statusBadge = document.getElementById('statusBadge');
    const statsBox = document.getElementById('statsBox');
    const latencyText = document.getElementById('latencyText');
    const countText = document.getElementById('countText');
    const tableContainer = document.getElementById('tableContainer');
    const detectionsList = document.getElementById('detectionsList');

    let currentFile = null;

    confSlider.addEventListener('input', (e) => {
      confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    dropArea.addEventListener('click', () => fileInput.click());

    dropArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('dragover'));

    dropArea.addEventListener('drop', (e) => {
      e.preventDefault();
      dropArea.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
      }
    });

    function handleFile(file) {
      currentFile = file;
      btnRun.disabled = false;
      const reader = new FileReader();
      reader.onload = (e) => {
        resultImage.src = e.target.result;
        resultImage.style.display = 'block';
        previewPlaceholder.style.display = 'none';
        statusBadge.style.display = 'none';
        tableContainer.style.display = 'none';
        statsBox.style.display = 'none';
      };
      reader.readAsDataURL(file);
    }

    // Quick samples
    document.querySelectorAll('.sample-chip').forEach(chip => {
      chip.addEventListener('click', async () => {
        const type = chip.getAttribute('data-sample');
        try {
          const res = await fetch(`/sample/${type}`);
          if (!res.ok) throw new Error("Failed to load sample");
          const blob = await res.blob();
          const file = new File([blob], `${type}.png`, { type: 'image/png' });
          handleFile(file);
          runDetection();
        } catch (err) {
          alert('Could not load sample: ' + err.message);
        }
      });
    });

    btnRun.addEventListener('click', runDetection);

    async function runDetection() {
      if (!currentFile) return;

      loader.style.display = 'block';
      resultImage.style.opacity = '0.3';
      btnRun.disabled = true;

      const formData = new FormData();
      formData.append('file', currentFile);
      formData.append('model', modelSelect.value);
      formData.append('conf', confSlider.value);

      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();

        loader.style.display = 'none';
        resultImage.style.opacity = '1.0';
        btnRun.disabled = false;

        if (data.error) {
          alert('Error: ' + data.error);
          return;
        }

        resultImage.src = 'data:image/jpeg;base64,' + data.annotated_image;
        resultImage.style.display = 'block';
        previewPlaceholder.style.display = 'none';

        // Update Status Badge
        statusBadge.style.display = 'inline-flex';
        if (data.detections.length === 0) {
          statusBadge.className = 'status-pill status-good';
          statusBadge.innerHTML = '● PASS (Defect Free)';
        } else {
          statusBadge.className = 'status-pill status-defect';
          statusBadge.innerHTML = `● FAIL (${data.detections.length} Defect${data.detections.length > 1 ? 's' : ''})`;
        }

        // Stats
        statsBox.style.display = 'flex';
        latencyText.textContent = `Inference: ${data.latency_ms.toFixed(1)} ms`;
        countText.textContent = `Detections: ${data.detections.length}`;

        // Table
        detectionsList.innerHTML = '';
        if (data.detections.length > 0) {
          tableContainer.style.display = 'block';
          data.detections.forEach(det => {
            const tr = document.createElement('tr');
            const color = det.color || '#3b82f6';
            tr.innerHTML = `
              <td><span class="tag" style="background: ${color}20; color: ${color}; border: 1px solid ${color}60;">${det.class}</span></td>
              <td>${(det.confidence * 100).toFixed(1)}%</td>
              <td style="font-family: monospace; font-size: 0.8rem; color: #cbd5e1;">[${det.box.map(v => Math.round(v)).join(', ')}]</td>
            `;
            detectionsList.appendChild(tr);
          });
        } else {
          tableContainer.style.display = 'none';
        }

      } catch (err) {
        loader.style.display = 'none';
        resultImage.style.opacity = '1.0';
        btnRun.disabled = false;
        alert('Prediction failed: ' + err.message);
      }
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/sample/<defect_type>")
def get_sample(defect_type):
    from flask import send_file
    # First look in standalone static/samples folder (for cloud deployments)
    static_sample = BASE_DIR / "static" / "samples" / f"{defect_type}.png"
    if static_sample.exists():
        return send_file(str(static_sample), mimetype="image/png")

    # Fallback to local metal_nut dataset if present
    if defect_type == "good":
        img_dir = BASE_DIR / "metal_nut" / "train" / "good"
    else:
        img_dir = BASE_DIR / "metal_nut" / "test" / defect_type
    
    candidates = sorted(img_dir.glob("*.png")) if img_dir.exists() else []
    if not candidates:
        return jsonify({"error": f"Sample for {defect_type} not found"}), 404
    return send_file(str(candidates[0]), mimetype="image/png")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    model_type = request.form.get("model", "320")
    conf_thresh = float(request.form.get("conf", 0.25))

    try:
        model = get_model(model_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    # Run inference & measure latency
    img_sz = 640 if model_type == "640" else 320
    t0 = time.perf_counter()
    results = model.predict(source=image, imgsz=img_sz, conf=conf_thresh, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000

    res = results[0]
    detections = []
    
    # Draw custom clear visual bounding boxes
    annotated = image.copy()
    names = res.names
    
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = names.get(cls_id, f"class_{cls_id}")
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        
        hex_color = CLASS_COLORS.get(cls_name, "#3b82f6")
        # hex to BGR
        h = hex_color.lstrip('#')
        bgr = tuple(int(h[i:i+2], 16) for i in (4, 2, 0))
        
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
        
        label = f"{cls_name} {conf*100:.1f}%"
        (w, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + w + 6, max(0, y1)), bgr, -1)
        cv2.putText(annotated, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "class": cls_name,
            "confidence": round(conf, 4),
            "box": xyxy,
            "color": hex_color
        })

    # Encode annotated image to JPEG base64
    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "detections": detections,
        "latency_ms": latency_ms,
        "annotated_image": img_b64
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
