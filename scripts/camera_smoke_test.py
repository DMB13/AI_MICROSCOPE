"""Camera smoke test: detect cameras 0-6, open first available, capture one frame.
Saves captured image to model/records/ and prints the path.
"""
import cv2
from pathlib import Path
import time

def detect_cameras(max_index=6):
    cams = []
    for i in range(max_index+1):
        cap = cv2.VideoCapture(i)
        if cap is None:
            continue
        if cap.isOpened():
            cams.append(i)
            cap.release()
    return cams


def capture_one(camera_index=0, timeout=5):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_index}")
    start = time.time()
    while True:
        ret, frame = cap.read()
        if ret:
            break
        if time.time() - start > timeout:
            cap.release()
            raise RuntimeError("Timeout waiting for camera frame")
    save_dir = Path(__file__).resolve().parents[1] / 'model' / 'records'
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = save_dir / f"smoke_capture_{timestamp}.png"
    cv2.imwrite(str(out), frame)
    cap.release()
    return out


def main():
    cams = detect_cameras()
    if not cams:
        print("No cameras detected (indices 0-6).")
        return
    print("Detected cameras:", cams)
    try:
        path = capture_one(cams[0])
        print("Captured frame saved to:", path)
    except Exception as e:
        print("Capture failed:", e)

if __name__ == '__main__':
    main()
