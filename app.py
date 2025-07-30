import cv2
import os
from ultralytics import YOLO
from datetime import datetime

# -------------------- Configuration --------------------
# Input source - Change this to a .mp4 path or a live stream URL (.m3u8)
VIDEO_SOURCE = "sample_video2.mp4"  # e.g., "http://video3.earthcam.com:1935/fecnetwork/9974.flv/chunklist.m3u8"
TOTAL_SEATS = 10             # Capacity for occupancy percentage
CONFIDENCE_THRESHOLD = 0.5   # YOLO confidence threshold
SAVE_SCREENSHOTS = True
SAVE_LOG = True

# Paths
OUTPUT_DIR = "outputs"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize model
model = YOLO("yolov8n.pt")

# Load video or stream
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise IOError("❌ Failed to open video source. Check the file path or stream URL.")

# Prepare log
log_data = []

print("✅ Starting detection... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > CONFIDENCE_THRESHOLD:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Calculate occupancy
    occupancy = (count / TOTAL_SEATS) * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cv2.putText(frame, f"People: {count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Occupancy: {occupancy:.1f}%", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Save annotated screenshot every 30 frames (~1 sec if video is 30fps)
    if SAVE_SCREENSHOTS and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
        img_name = f"{OUTPUT_DIR}/frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(img_name, frame)

    # Log entry
    if SAVE_LOG:
        log_data.append(f"{timestamp}, Count: {count}, Occupancy: {occupancy:.2f}%")

    # Display
    cv2.imshow("People Detection Monitor", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save log report
if SAVE_LOG:
    log_path = os.path.join(OUTPUT_DIR, "occupancy_log.txt")
    with open(log_path, "w") as f:
        f.write("Timestamp, People Count, Occupancy\n")
        for line in log_data:
            f.write(line + "\n")

print(f"\n📸 Screenshots saved to: {OUTPUT_DIR}")
print(f"📄 Log file saved to: {log_path}")
