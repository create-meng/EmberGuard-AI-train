import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# Load models
yolo_model = YOLO(r"")
lstm_model = load_model(r"")

# Paths
video_path = r""

# Parameters
SEQ_LEN = 30
features_buffer = []

# Try to infer number of classes from model output
# Run a dummy prediction to get output shape
dummy_seq = np.zeros((1, SEQ_LEN, 8))  # 8 features per frame
pred = lstm_model.predict(dummy_seq, verbose=0)
num_classes = pred.shape[1]

# Define labels (edit this list to match your training dataset)
# If not enough labels, fill with placeholders
user_labels = ["no_fire", "smoke", "fire"]  # <-- CHANGE if your dataset has more classes
if len(user_labels) < num_classes:
    labels = user_labels + [f"class{i}" for i in range(len(user_labels), num_classes)]
else:
    labels = user_labels[:num_classes]

print(f"[INFO] Model expects {num_classes} classes → Using labels: {labels}")

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Collect features
            features = [cx, cy, w, h, area, aspect_ratio, conf, cls]
            features_buffer.append(features)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"YOLO cls:{cls} conf:{conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # Once we have 30 frames, predict with LSTM
    if len(features_buffer) >= SEQ_LEN:
        seq = np.array(features_buffer[-SEQ_LEN:])  # last 30 frames
        seq = np.expand_dims(seq, axis=0)  # shape (1, 30, features)

        pred = lstm_model.predict(seq, verbose=0)
        label = int(np.argmax(pred))

        pred_label = labels[label]
        prob = float(np.max(pred))

        # Overlay prediction on video
        cv2.putText(frame, f"🔥 Prediction: {pred_label} ({prob:.2f})",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)

    # Show video window
    cv2.imshow("Fire Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
