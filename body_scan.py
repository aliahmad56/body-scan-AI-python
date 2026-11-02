import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)  # Set static_image_mode to True for processing a single image
mp_drawing = mp.solutions.drawing_utils

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return image

def scan_the_body(reference_image_path):
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

    # Function to calculate arm length in pixels
    def calculate_arm_length(landmarks):
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        arm_length = np.sqrt((left_shoulder.x - left_wrist.x) ** 2 + (left_shoulder.y - left_wrist.y) ** 2)
        return arm_length

    # Function to calculate leg length in pixels
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
    reference_waist_cm = 93.98 # Actual waist size in cm
    reference_shoulder_cm = 45.72  # Actual shoulder size in cm
    reference_arm_cm = 73.66  # Actual arm length in cm
    reference_leg_cm = 96.52  # Actual leg length in cm
    reference_chest_cm = 93.98  # Actual chest size in cm

    try:
        # Load reference image
        reference_image = load_image("reference_image.jpeg")
        reference_results = pose.process(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))

        if not reference_results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the reference image.")

        reference_waist_pixels = calculate_waist_width(reference_results.pose_landmarks.landmark)
        reference_shoulder_pixels = calculate_shoulder_width(reference_results.pose_landmarks.landmark)
        reference_arm_pixels = calculate_arm_length(reference_results.pose_landmarks.landmark)
        reference_leg_pixels = calculate_leg_length(reference_results.pose_landmarks.landmark)
        reference_chest_pixels = calculate_chest_width(reference_results.pose_landmarks.landmark)

        # Load input image
        input_image_path = reference_image_path  # Path to the input image
        input_image = load_image(input_image_path)
        results = pose.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            raise ValueError("No pose landmarks detected in the input image.")

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

        return waist_width_cm, arm_length_cm, leg_length_cm, shoulder_width_cm, chest_width_cm

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

'''# Example usage
reference_image_path = "test.jpeg"
results = scan_the_body(reference_image_path)

if results:
    waist, arm, leg, shoulder, chest = results
    print(f"Waist: {waist:.2f} cm")
    print(f"Arm length: {arm:.2f} cm")
    print(f"Leg length: {leg:.2f} cm")
    print(f"Shoulder width: {shoulder:.2f} cm")
    print(f"Chest width: {chest:.2f} cm")
else:
    print("Failed to process the images.")'''