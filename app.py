from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import math

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize rep count, set counter, and state
rep_count = 0
setCounter = 0
arm_raised = False

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    radians = math.atan2(a[1] - b[1], a[0] - b[0]) - math.atan2(c[1] - b[1], c[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# Function to update rep count based on angle
def update_rep_count(rep_count, angle, arm_raised, setCounter, maxAngle, minAngle):
    if angle >= maxAngle:
        arm_raised = True
    if arm_raised and angle <= minAngle:
        rep_count += 1
        arm_raised = False

    if rep_count == 10:
        setCounter += 1
        rep_count = 0  
    return rep_count, arm_raised, setCounter

def choosePoints(indicator, results):
    landmarks = results.pose_landmarks.landmark
    if indicator == 0:  # Bicep curl
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        return shoulder, elbow, wrist
    elif indicator == 1:  # Shoulder press
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        return shoulder, elbow, wrist
    elif indicator == 2:  # Lateral raise
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        return elbow, shoulder, right_hip

def gen(minAngle, maxAngle, indicator):
    global rep_count, setCounter, arm_raised

    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                point1, point2, point3 = choosePoints(indicator, results)

                # Calculate angle
                angle =calculate_angle(point1,point2,point3)

                # Display angle
                cv2.putText(frame, str(angle), 
                            (int(point2.x * frame.shape[1]), int(point2.y * frame.shape[0])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                           )

                # Update and display rep count and set counter
                rep_count, arm_raised, setCounter = update_rep_count(rep_count, angle, arm_raised, setCounter, maxAngle, minAngle)
                cv2.putText(frame, f'Rep Count: {rep_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Set Count: {setCounter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/push')
def push():
    return render_template('push.html')

@app.route('/pull')
def pull():
    return render_template('pull.html')

@app.route('/leg')
def leg():
    return render_template('leg.html')

@app.route('/bicep_curl')
def bicep_curl():
    return render_template('bicepweb.html')

@app.route('/bicep_curl_feed')
def bicep_curl_feed():
    return Response(gen(15, 173, 0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shoulder_press')
def shoulder_press():
    return render_template('shoulderPress.html')

@app.route('/shoulder_press_feed')
def shoulder_press_feed():
    return Response(gen(15, 170, 1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/lateral_raise')
def lateral_raise():
    return render_template('lateralRaise.html')

@app.route('/lateral_raise_feed')
def lateral_raise_feed():
    return Response(gen(20, 80, 2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)