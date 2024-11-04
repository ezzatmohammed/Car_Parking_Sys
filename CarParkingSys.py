import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from collections import deque

# Configuration
CONFIG = {
    'CONFIDENCE_THRESHOLD': 0.4,  # Minimum confidence for car detection
    'HISTORY_FRAMES': 5,  # Number of frames to keep in history
    'OCCUPANCY_THRESHOLD': 0.6,  # Percentage of frames needed to change state
    'OVERLAP_THRESHOLD': 0.3  # Minimum overlap required to consider spot occupied
}

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video file
cap = cv2.VideoCapture('easy1.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original dimensions and calculate new size
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_width = original_width // 2
new_height = original_height // 2

# Load saved ROIs
try:
    with open("CarParkROIs", 'rb') as f:
        roi_list = pickle.load(f)
except:
    roi_list = []
    print("No saved parking spots found. Please draw spots manually.")

# Initialize spot history
spot_history = {i: deque(maxlen=CONFIG['HISTORY_FRAMES']) for i in range(len(roi_list))}

# Variables for drawing
drawing = False
current_roi = []


def calculate_overlap(roi_mask, car_box):
    # Ensure car_box coordinates are within roi_mask bounds
    x1 = max(0, car_box[0])
    y1 = max(0, car_box[1])
    x2 = min(roi_mask.shape[1], car_box[2])
    y2 = min(roi_mask.shape[0], car_box[3])

    # Create car mask
    car_mask = np.zeros_like(roi_mask, dtype=np.uint8)
    cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)

    # Calculate intersection
    intersection = cv2.bitwise_and(roi_mask, car_mask)
    roi_area = cv2.countNonZero(roi_mask)

    # Handle zero area case
    if roi_area == 0:
        return 0.0

    # Calculate and return overlap ratio
    overlap_ratio = cv2.countNonZero(intersection) / roi_area
    return float(overlap_ratio)


def mouseclick(event, x, y, flags, param):
    global drawing, current_roi, roi_list, spot_history

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_roi = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_roi.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(current_roi) > 2:  # Only add if we have enough points
            roi_list.append(current_roi.copy())
            spot_history[len(roi_list) - 1] = deque(maxlen=CONFIG['HISTORY_FRAMES'])
            current_roi = []
            # Save ROIs to file
            with open("CarParkROIs", 'wb') as f:
                pickle.dump(roi_list, f)

    elif event == cv2.EVENT_RBUTTONDOWN:
        click_point = np.array([[x, y]], dtype=np.int32)
        for i, roi in enumerate(roi_list):
            if cv2.pointPolygonTest(np.array(roi), (x, y), False) >= 0:
                roi_list.pop(i)
                spot_history.pop(i)
                # Reindex remaining spots
                new_history = {}
                for new_i in range(len(roi_list)):
                    if new_i < i:
                        new_history[new_i] = spot_history[new_i]
                    else:
                        new_history[new_i] = spot_history[new_i + 1]
                spot_history = new_history
                # Save updated ROIs
                with open("CarParkROIs", 'wb') as f:
                    pickle.dump(roi_list, f)
                break


cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouseclick)

while True:
    ret, frame = cap.read()
    if not ret:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Run YOLOv8 detection
    results = model(resized_frame, stream=True)

    detected_cars = []
    for result in results:
        for box in result.boxes:
            if box.cls[0] == 2 and box.conf[0] >= CONFIG['CONFIDENCE_THRESHOLD']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_cars.append((x1, y1, x2, y2))

    free_spots = 0
    occupied_spots = 0

    # Process each ROI
    for idx, roi_points in enumerate(roi_list):
        # Create mask for this ROI
        roi_mask = np.zeros(resized_frame.shape[:2], dtype=np.uint8)
        roi_np = np.array(roi_points)
        cv2.fillPoly(roi_mask, [roi_np], 255)

        # Check overlap with detected cars
        spot_occupied = False
        for car_box in detected_cars:
            overlap = calculate_overlap(roi_mask, car_box)
            if overlap > CONFIG['OVERLAP_THRESHOLD']:
                spot_occupied = True
                break

        # Update history
        spot_history[idx].append(spot_occupied)

        # Calculate smoothed state
        occupation_ratio = sum(spot_history[idx]) / len(spot_history[idx])
        is_occupied = occupation_ratio >= CONFIG['OCCUPANCY_THRESHOLD']

        # Draw ROI
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        cv2.polylines(resized_frame, [roi_np], True, color, 2)

        # Add occupation ratio text
        moments = cv2.moments(roi_np)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            ratio_text = f"{occupation_ratio:.2f}"
            cv2.putText(resized_frame, ratio_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if is_occupied:
            occupied_spots += 1
        else:
            free_spots += 1

    # Draw current ROI while drawing
    if drawing and current_roi:
        points = np.array(current_roi, dtype=np.int32)
        cv2.polylines(resized_frame, [points], False, (255, 255, 0), 2)

    # Display counts
    cv2.putText(resized_frame, f'Free: {free_spots}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(resized_frame, f'Occupied: {occupied_spots}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(resized_frame, f'Total: {len(roi_list)}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
