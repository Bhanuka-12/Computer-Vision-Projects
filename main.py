import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing styles
mp_draw = mp.solutions.drawing_utils

# Define the tip landmarks for each finger
finger_tips = [4, 8, 12, 16, 20]

def find_hands(img, draw=True):
    """Detect hands in the image and draw landmarks."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img, results

def find_position(img, results, hand_no=0, draw=True):
    """Find the position of landmarks on the hand."""
    lm_list = []
    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[hand_no]
        for id, lm in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
    return lm_list

def count_fingers(landmarks):
    """Count the number of fingers raised."""
    fingers_status = []
    
    # Detect thumb: compare x-coordinates for open/closed status
    if landmarks[finger_tips[0]][1] > landmarks[finger_tips[0] - 1][1]:
        fingers_status.append(1)
    else:
        fingers_status.append(0)
        
    # Detect the rest of the fingers: compare y-coordinates for open/closed status
    for finger in range(1, 5):
        if landmarks[finger_tips[finger]][2] < landmarks[finger_tips[finger] - 2][2]:
            fingers_status.append(1)
        else:
            fingers_status.append(0)
            
    return fingers_status.count(1)

# Capture video from the default camera (laptop's webcam)
video_capture = cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Add a small delay to ensure the camera is initialized
time.sleep(2)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    success, frame = video_capture.read()
    
    if not success:
        print("Error: Failed to read frame from video capture.")
        break
    
    # Detect hands in the frame
    frame, results = find_hands(frame)
    
    # Find landmark positions
    landmarks = find_position(frame, results, draw=False)
    
    if len(landmarks) != 0:
        # Count fingers
        total_fingers_up = count_fingers(landmarks)
        
        # Display the result on the video frame
        cv2.rectangle(frame, (10, 10), (100, 70), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, str(total_fingers_up), (34, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    
    # Write the frame to the output file
    out.write(frame)
    
    # Show the video frame
    cv2.imshow("Finger Counter", frame)
    
    # Exit if 'q' or 'backspace' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 8:  # 8 is the ASCII code for backspace
        break

# Release resources
video_capture.release()
out.release()
cv2.destroyAllWindows()
