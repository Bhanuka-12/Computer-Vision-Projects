import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Initialize pose estimator
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video
cap = cv2.VideoCapture('D:\Academic\Others\CV Projects\Javeline_throw\Data\FT5-GT.mp4')

# Define optimal release parameters
optimal_release_velocity = 24.3  # m/s
optimal_release_angle = 40.7  # degrees

# Define optimal physical fitness metrics
optimal_60m_sprint = 4.25  # seconds
optimal_pullups = 12.94  # number

# Initialize deque for smoothing angle values
angle_deque = deque(maxlen=5)
velocity_deque = deque(maxlen=5)

# Initialize variables to track the previous wrist position
prev_wrist_y = None
prev_wrist_x = None
prev_wrist_time = None
throw_detected = False

def calculate_velocity(prev_wrist_x, prev_wrist_y, prev_wrist_time, wrist_x, wrist_y, current_time):
    if prev_wrist_time is None:
        return 0.0
    distance = math.sqrt((wrist_x - prev_wrist_x)**2 + (wrist_y - prev_wrist_y)**2)
    time_elapsed = current_time - prev_wrist_time
    velocity = distance / time_elapsed
    return velocity

def calculate_angle(shoulder, elbow, wrist):
    shoulder_elbow = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y])
    elbow_wrist = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    
    dot_product = np.dot(shoulder_elbow, elbow_wrist)
    magnitude = np.linalg.norm(shoulder_elbow) * np.linalg.norm(elbow_wrist)
    
    angle = math.degrees(math.acos(dot_product / magnitude))
    return angle

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract specific landmarks for release parameters
        shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds
        
        # Detect throw based on wrist y-coordinate change
        if prev_wrist_y is not None and abs(wrist.y - prev_wrist_y) > 0.1:
            throw_detected = True
        
        if throw_detected:
            # Calculate release velocity and angle
            release_velocity = calculate_velocity(prev_wrist_x, prev_wrist_y, prev_wrist_time, wrist.x, wrist.y, current_time)
            release_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Smooth the angle and velocity using a moving average
            angle_deque.append(release_angle)
            smoothed_angle = np.mean(angle_deque)
            
            velocity_deque.append(release_velocity)
            smoothed_velocity = np.mean(velocity_deque)
            
            # Compare against optimal values and provide feedback
            feedback = "Pose Correct"
            if smoothed_velocity < optimal_release_velocity or abs(smoothed_angle - optimal_release_angle) > 5:
                feedback = "Pose Incorrect"
            
            cv2.putText(frame, f"Release Velocity: {smoothed_velocity:.2f} m/s (Optimal: {optimal_release_velocity} m/s)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Release Angle: {smoothed_angle:.2f} degrees (Optimal: {optimal_release_angle} degrees)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, feedback, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if feedback == "Pose Correct" else (0, 0, 255), 1)
            
            # Display feedback on physical fitness metrics
            cv2.putText(frame, f"60m Sprint: {optimal_60m_sprint} seconds", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Pull-ups: {optimal_pullups} reps", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        prev_wrist_x = wrist.x
        prev_wrist_y = wrist.y
        prev_wrist_time = current_time
    
    # Write the frame to the output video
    out.write(frame)
    
    cv2.imshow('Javelin Throw Analysis', frame)
        
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
