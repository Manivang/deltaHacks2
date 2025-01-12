import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize rep count and state
rep_count = 0
arm_raised = False

# Function to calculate angle between three points
def calculate_angle(rshoulder, rHip, relbow):
    rshoulder = [rshoulder.x, rshoulder.y]
    rHip = [rHip.x, rHip.y]
    relbow = [relbow.x, relbow.y]
    radians = math.atan2(relbow[1] - rshoulder[1], relbow[0] - rshoulder[0]) - math.atan2(rHip[1] - rshoulder[1], rHip[0] - rshoulder[0])
    angle = abs(radians * 180.0 / math.pi)
    return angle

# Function to update rep count based on angle
def update_rep_count(rep_count, angle, arm_raised):
    if angle >= 80:
        arm_raised = True
    if arm_raised and angle < 20:
        rep_count += 1
        arm_raised = False
    return rep_count, arm_raised

# Initialize video capture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find pose
        results = pose.process(image)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for shoulder, elbow, and hip
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Calculate angle
            angle = calculate_angle(shoulder, right_hip, elbow)

            # Display angle
            cv2.putText(frame, str(angle), 
                        (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                       )

            # Update and display rep count
            rep_count, arm_raised = update_rep_count(rep_count, angle, arm_raised)
            cv2.putText(frame, f'Rep Count: {rep_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Lateral Raises', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()