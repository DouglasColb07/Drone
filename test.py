import cv2
import mediapipe as mp
from codrone_edu.drone import Drone
import time
import math

# -------------------------------
# STEP 1: INITIALIZE DRONE AND CAMERA
# -------------------------------
drone = Drone()
drone.pair()
print("Paired successfully!")

# Initialize camera and MediaPipe Hand tracking
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# STEP 2: DEFINE MOVEMENT FUNCTIONS
# -------------------------------

def move_forward(distance):
    """ Moves the drone forward based on distance """
    print(f"🚁 Moving forward by {distance} meters")
    drone.set_pitch(30)  # Tilt forward
    drone.move(distance)
    time.sleep(2)  # Allow time for the drone to move

def turn_left(distance):
    """ Turns the drone left based on distance """
    print(f"↺ Turning left by {distance} meters")
    drone.set_yaw(-30)   # Rotate left
    drone.move(distance)
    time.sleep(2)  # Allow time for the drone to turn

def turn_right(distance):
    """ Turns the drone right based on distance """
    print(f"↻ Turning right by {distance} meters")
    drone.set_yaw(30)    # Rotate right
    drone.move(distance)
    time.sleep(2)  # Allow time for the drone to turn

# -------------------------------
# STEP 3: HAND DETECTION AND MOVEMENT LOGIC
# -------------------------------

print("Taking off...")
drone.takeoff()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for easier gesture control
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark coordinates for the thumb and index finger
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]

            # Calculate the Euclidean distance between thumb and index finger
            x_diff = thumb_tip.x - index_tip.x
            y_diff = thumb_tip.y - index_tip.y
            z_diff = thumb_tip.z - index_tip.z

            distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)  # 3D Euclidean distance

            print(f"Distance: {distance}")  # Debugging print

            # Convert distance to a movement range (scale it to a reasonable distance for drone)
            move_distance = distance * 10  # You can tweak the multiplier as needed for larger/smaller movements

            # Drone movement based on finger gestures
            if x_diff > 0.1:
                turn_left(move_distance)  # Move left when thumb is to the right of the index finger
            elif x_diff < -0.1:
                turn_right(move_distance)  # Move right when thumb is to the left of the index finger
            elif y_diff < -0.1:
                move_forward(move_distance)  # Move forward when the index finger moves upward

    # Display the camera feed
    cv2.imshow("Drone Control with Hand Gestures", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# STEP 4: LAND AND CLEAN UP
# -------------------------------
print("Landing...")
drone.land()
drone.close()

cap.release()
cv2.destroyAllWindows()

print("Program complete ✅")
