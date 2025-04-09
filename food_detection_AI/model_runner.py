##### Calorie Counter (Enhanced with Intake Limits + Burn Time Estimation) #####

import os
import sys
import cv2
import csv
import time
import argparse
from datetime import datetime
from ultralytics import YOLO

# -------- Argument Parser --------
parser = argparse.ArgumentParser(description="Real-time YOLO Food Detection")
parser.add_argument('--model', type=str, default='food_detection_model.pt', help='Path to YOLO model (.pt)')
parser.add_argument('--cam', type=int, default=0, help='Camera index (default 0)')
parser.add_argument('--record', action='store_true', help='Record result video')
args = parser.parse_args()

# -------- Configurable Parameters --------
model_path = args.model
cam_index = args.cam
record = args.record
imgW, imgH = 1280, 720
min_thresh = 0.50

# -------- Nutrition Info Dictionary --------
nutrition_info = {
    'Dairy Milk Hazelnut': [83, 7.4],
    'Maggi Sup Ayam': [254, 2.2],
    'Milo Nuggets': [74, 6.8],
    'Mister Potato Crips Original': [139, 0.3],
    'Pocky Double Choco': [77, 13],
    'Samyang Spicy Noodle': [425, 6],
    'Tiger Susu Biscuits': [157, 7.3],
    'Tropicana Twister Orange Juice': [64, 14.5],
    'Twiggies Cream Dream Vanila Bread': [150, 10.9],
    'Wonda Latte Milk Coffee Drink': [86, 10.3]
}

# -------- Intake Limits --------
calorie_limit = 600        # Max calories per sitting
sugar_limit = 25.0         # Max sugar (g) per sitting

# -------- Model Check --------
if not os.path.exists(model_path):
    print(f"Model file not found at: {model_path}")
    sys.exit()

print(f"Using model: {model_path}")
model = YOLO(model_path, task='detect')
labels = model.names

# -------- Camera Init --------
cap = cv2.VideoCapture(cam_index)
cap.set(3, imgW)
cap.set(4, imgH)

# -------- Recording --------
if record:
    record_name = f'detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), 30, (imgW, imgH))

# -------- Logging --------
log_file = open('detection_log.csv', 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Timestamp', 'Detected Items', 'Total Calories', 'Total Sugar (g)', 'Status'])

# -------- Colors --------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# -------- Main Loop --------
prev_time = time.time()
cv2.namedWindow("Food detection results", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera disconnected or not working.")
        break

    results = model.track(frame, verbose=False)
    detections = results[0].boxes
    foods_detected = []

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, label_size[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), 
                          (xmin + label_size[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            foods_detected.append(classname)

    # -------- Nutrition Calculation --------
    total_calories = 0
    total_sugar = 0
    for food in foods_detected:
        if food in nutrition_info:
            calories, sugar = nutrition_info[food]
            total_calories += calories
            total_sugar += sugar
        else:
            print(f"[Warning] '{food}' not found in nutrition dictionary.")

    # -------- Status Check --------
    if total_calories <= calorie_limit and total_sugar <= sugar_limit:
        status = "SAFE"
        status_color = (0, 255, 0)
    elif total_calories > calorie_limit and total_sugar > sugar_limit:
        status = "CALORIES & SUGAR EXCEEDED"
        status_color = (0, 0, 255)
    elif total_calories > calorie_limit:
        status = "CALORIES EXCEEDED"
        status_color = (0, 0, 255)
    elif total_sugar > sugar_limit:
        status = "SUGAR EXCEEDED"
        status_color = (0, 165, 255)

    # -------- FPS --------
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # -------- Overlay --------
    cv2.rectangle(frame, (10, 10), (500, 190), (50, 50, 50), cv2.FILLED)
    cv2.putText(frame, f'Number of food: {len(foods_detected)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,102,51), 2)
    cv2.putText(frame, f'Total calories: {total_calories:.1f} / {calorie_limit}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (51,204,51), 2)
    cv2.putText(frame, f'Total sugar (g): {total_sugar:.1f} / {sugar_limit}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,204,255), 2)
    cv2.putText(frame, f'STATUS: {status}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f'FPS: {fps:.2f}', (350, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # -------- Burn Time Calculation --------
    burn_rate_per_min = 10  # kcal per min
    sugar_kcal_per_gram = 4

    calorie_burn_time = total_calories / burn_rate_per_min
    sugar_burn_time = (total_sugar * sugar_kcal_per_gram) / burn_rate_per_min

    # -------- Burn Time Box (Top-right) --------
    box_x1 = imgW - 310
    box_y1 = 10
    box_x2 = imgW - 10
    box_y2 = 110

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), cv2.FILLED)
    cv2.putText(frame, 'Burn Estimation', (box_x1 + 10, box_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Calories: {calorie_burn_time:.1f} min', (box_x1 + 10, box_y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 204, 51), 2)
    cv2.putText(frame, f'Sugar: {sugar_burn_time:.1f} min', (box_x1 + 10, box_y1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 204, 255), 2)

    # -------- User Instructions (Bottom-left) --------
    instruction_y_start = imgH - 70
    cv2.putText(frame, "Press 'q' to Quit", (20, instruction_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 's' to Pause", (20, instruction_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'p' to Capture", (20, instruction_y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # -------- Display --------
    cv2.imshow("Food detection results", frame)

    if record:
        recorder.write(frame)

    # -------- Log Entry --------
    if foods_detected:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv_writer.writerow([timestamp, ', '.join(foods_detected), total_calories, total_sugar, status])

    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey()
    elif key == ord('p'):
        cv2.imwrite(f'capture_{datetime.now().strftime("%H%M%S")}.png', frame)

# -------- Cleanup --------
cap.release()
if record:
    recorder.release()
log_file.close()
cv2.destroyAllWindows()
