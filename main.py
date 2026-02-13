import cv2
import time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# =========================
# MODEL SETUP
# =========================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose_solution = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_selfie_segmentation = mp.solutions.selfie_segmentation

selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1
)


model = YOLO("yolov8n.pt")

# =========================
# GLOBAL VARIABLES
# =========================

mode = "NORMAL"
filter_type = "GRAY"
activity = "NONE"
rep_stage = "-"
reps = 0
tracker = None
tracking = False

MODES = ["NORMAL", "FILTER", "SEGMENT", "DETECT", "CLASSIFY"]
FILTERS = ["GRAY", "BLUR", "EDGE", "SHARP"]

# =========================
# ANGLE FUNCTION
# =========================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360-angle if angle > 180 else angle

# =========================
# MOUSE CLICK
# =========================

def mouse_click(event, x, y, flags, param):
    global mode, filter_type, reps, rep_stage, tracking

    if event == cv2.EVENT_LBUTTONDOWN:

        # Top bar dimensions MUST match UI
        button_width = 110
        button_spacing = 10
        start_x = 10

        for i, name in enumerate(MODES):
            x1 = start_x + i * (button_width + button_spacing)
            x2 = x1 + button_width

            if x1 < x < x2 and 10 < y < 50:
                mode = name
                reps = 0
                rep_stage = "-"
                tracking = False

        # Filter buttons
        if mode == "FILTER":
            for i, name in enumerate(FILTERS):
                x1 = 20 + i*160
                x2 = x1 + 140
                if x1 < x < x2 and 70 < y < 110:
                    filter_type = name


cv2.namedWindow("AI Fitness Vision")
cv2.setMouseCallback("AI Fitness Vision", mouse_click)

cap = cv2.VideoCapture(0)
prev_time = 0
print("Before segmenter")



while True:
        ret, frame = cap.read()
        if not ret:
            break

        activity = "NONE"
        output = frame.copy()
        h, w, _ = frame.shape


        # =========================
        # NORMAL MODE (POSTURE) — EARS + SHOULDERS ONLY
        # =========================
        if mode == "NORMAL":

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results_pose = pose_solution.process(image_rgb)
            image_rgb.flags.writeable = True

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                # Get both ears and shoulders
                left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                # Average points (SAME as simple script)
                avg_ear = [
                    (left_ear[0] + right_ear[0]) / 2,
                    (left_ear[1] + right_ear[1]) / 2
                ]

                avg_shoulder = [
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2
                ]

                h, w, _ = output.shape
                ear_point = tuple(np.multiply(avg_ear, [w, h]).astype(int))
                shoulder_point = tuple(np.multiply(avg_shoulder, [w, h]).astype(int))

                # -------------------------
                # POSTURE LOGIC (EXACT SAME)
                # -------------------------
                dx = abs(ear_point[0] - shoulder_point[0])
                threshold = 20   # EXACT SAME

                if dx > threshold:
                    posture_status = "BAD POSTURE"
                    color = (0, 0, 255)
                else:
                    posture_status = "GOOD POSTURE"
                    color = (0, 255, 0)

                # Draw and display (EXACT SAME)
                cv2.line(output, ear_point, shoulder_point, color, 3)

                cv2.putText(output, posture_status, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                cv2.putText(output, f"dx: {dx}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    output,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                activity = "POSTURE MONITORING"


        # =========================
        # FILTER MODE
        # =========================
        elif mode == "FILTER":

            if filter_type == "GRAY":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            elif filter_type == "BLUR":
                output = cv2.GaussianBlur(frame, (15,15), 0)

            elif filter_type == "EDGE":
                edges = cv2.Canny(frame, 50, 150)
                output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            elif filter_type == "SHARP":
                kernel = np.array([[0,-1,0],
                                   [-1,5,-1],
                                   [0,-1,0]])
                output = cv2.filter2D(frame, -1, kernel)

        # =========================
        # SEGMENT MODE
        # =========================
        elif mode == "SEGMENT":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_seg = selfie_segmentation.process(rgb)

            if results_seg.segmentation_mask is not None:
                mask = results_seg.segmentation_mask

                blurred = cv2.GaussianBlur(frame, (55, 55), 0)

                condition = mask > 0.5
                condition = np.stack((condition,)*3, axis=-1)

                output = np.where(condition, frame, blurred)
            else:
                output = frame

        # =========================
        # DETECT + TRACK
        # =========================
        elif mode == "DETECT":

            if not tracking:
                results = model(frame)
                for r in results:
                    for box in r.boxes:
                        x1,y1,x2,y2 = box.xyxy[0]
                        cls = int(box.cls[0])
                        if model.names[cls] == "person":
                            bbox = (int(x1), int(y1),
                                    int(x2-x1), int(y2-y1))
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, bbox)
                            tracking = True
                            break
            else:
                success, bbox = tracker.update(frame)
                if success:
                    x,y,w_box,h_box = [int(v) for v in bbox]
                    cv2.rectangle(output,(x,y),
                                  (x+w_box,y+h_box),
                                  (0,255,0),2)

        # =========================
        # CLASSIFY
        # =========================
        elif mode == "CLASSIFY":

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results_pose = pose_solution.process(image_rgb)
            image_rgb.flags.writeable = True

            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark

                # LEFT ARM
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Curl counter logic
                if angle > 150:
                    rep_stage = "DOWN"

                if angle < 70 and rep_stage == "DOWN":
                    rep_stage = "UP"
                    reps += 1


                activity = "BICEP CURL"

                # Draw angle
                cv2.putText(output,
                            f"Angle: {int(angle)}",
                            tuple(np.multiply(elbow, [w, h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,255,255), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    output,
                    results_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

        # =========================
        # FPS
        # =========================
        current_time = time.time()
        fps = 1/(current_time-prev_time) if prev_time else 0
        prev_time = current_time

        # =========================
        # TOP BAR
        # =========================
        cv2.rectangle(output, (0, 0), (w, 60), (40, 40, 40), -1)

        button_width = 110
        button_spacing = 10
        start_x = 10

        for i, name in enumerate(MODES):
            x1 = start_x + i * (button_width + button_spacing)
            x2 = x1 + button_width

            color = (0, 200, 0) if mode == name else (90, 90, 90)

            cv2.rectangle(output, (x1, 10), (x2, 50), color, -1)

            # Center text inside button
            text_size = cv2.getTextSize(name,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, 1)[0]

            text_x = x1 + (button_width - text_size[0]) // 2
            text_y = 35

            cv2.putText(output, name,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)


        # =========================
        # FILTER OPTIONS BAR
        # =========================
        if mode == "FILTER":
            cv2.rectangle(output,(0,60),(w,120),(60,60,60),-1)
            for i, name in enumerate(FILTERS):
                x1 = 20 + i*160
                color = (0,150,255) if filter_type == name else (100,100,100)
                cv2.rectangle(output,(x1,70),(x1+140,110),color,-1)
                cv2.putText(output,name,
                            (x1+30,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(255,255,255),2)

        # =========================
        # BOTTOM STATS BAR
        # =========================
        cv2.rectangle(output,(0,h-80),(w,h),(30,30,30),-1)

        cv2.putText(output,f"Exercise: {activity}",
                    (20,h-45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,255,255),2)

        cv2.putText(output,f"Reps: {reps}",
                    (300,h-45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

        cv2.putText(output,f"Stage: {rep_stage}",
                    (500,h-45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,255,255),2)

        cv2.putText(output,f"FPS: {int(fps)}",
                    (700,h-45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(200,200,200),2)

        cv2.imshow("AI Fitness Vision", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()