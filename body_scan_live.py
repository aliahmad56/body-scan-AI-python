import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate waist width in pixels
def calculate_waist_width(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    waist_width = np.sqrt((left_hip.x - right_hip.x) ** 2 + (left_hip.y - right_hip.y) ** 2)
    return waist_width

# Function to calculate shoulder width in pixels
def calculate_shoulder_width(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
    return shoulder_width

# Function to calculate arm's length in pixels
def calculate_arm_length(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    arm_length = np.sqrt((left_shoulder.x - left_wrist.x) ** 2 + (left_shoulder.y - left_wrist.y) ** 2)
    return arm_length

# Function to calculate leg's length in pixels
def calculate_leg_length(landmarks):
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    leg_length = np.sqrt((left_hip.x - left_ankle.x) ** 2 + (left_hip.y - left_ankle.y) ** 2)
    return leg_length

# Function to calculate chest width in pixels
def calculate_chest_width(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    chest_width = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
    return chest_width

# Reference image dimensions and actual sizes in cm
reference_image_path = "model.jpg"
reference_waist_cm = 76.2  # Actual waist size in cm
reference_shoulder_cm = 49  # Actual shoulder size in cm
reference_arm_cm = 55  # Actual arm length in cm
reference_leg_cm = 95  # Actual leg length in cm
reference_chest_cm = 100  # Actual chest size in cm

reference_image = cv2.imread(reference_image_path)
reference_results = pose.process(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))

if reference_results.pose_landmarks:
    reference_waist_pixels = calculate_waist_width(reference_results.pose_landmarks.landmark)
    reference_shoulder_pixels = calculate_shoulder_width(reference_results.pose_landmarks.landmark)
    reference_arm_pixels = calculate_arm_length(reference_results.pose_landmarks.landmark)
    reference_leg_pixels = calculate_leg_length(reference_results.pose_landmarks.landmark)
    reference_chest_pixels = calculate_chest_width(reference_results.pose_landmarks.landmark)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Calculate measurements in pixels
        waist_width_pixels = calculate_waist_width(landmarks)
        shoulder_width_pixels = calculate_shoulder_width(landmarks)
        arm_length_pixels = calculate_arm_length(landmarks)
        leg_length_pixels = calculate_leg_length(landmarks)
        chest_width_pixels = calculate_chest_width(landmarks)

        # Convert pixels to centimeters
        waist_width_cm = (waist_width_pixels / reference_waist_pixels) * reference_waist_cm
        shoulder_width_cm = (shoulder_width_pixels / reference_shoulder_pixels) * reference_shoulder_cm
        arm_length_cm = (arm_length_pixels / reference_arm_pixels) * reference_arm_cm
        leg_length_cm = (leg_length_pixels / reference_leg_pixels) * reference_leg_cm
        chest_width_cm = (chest_width_pixels / reference_chest_pixels) * reference_chest_cm

        # Draw landmarks and reference lines
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1]), 
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1]), 
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]))
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]), 
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]), 
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))

        cv2.line(frame, left_hip, right_hip, (0, 255, 0), 2)
        cv2.line(frame, left_shoulder, right_shoulder, (255, 0, 0), 2)
        cv2.putText(frame, f'Waist: {waist_width_cm:.2f} cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Shoulder: {shoulder_width_cm:.2f} cm', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Arm Length: {arm_length_cm:.2f} cm', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Leg Length: {leg_length_cm:.2f} cm', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f'Chest: {chest_width_cm:.2f} cm', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    cv2.imshow('Body Measurements', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
