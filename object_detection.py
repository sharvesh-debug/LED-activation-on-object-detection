import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO pins for LEDs
LED_ONE_FINGER = 17    # GPIO pin for one finger detection LED
LED_TWO_FINGERS = 27   # GPIO pin for two fingers detection LED
LED_THREE_FINGERS = 22 # GPIO pin for three fingers detection LED

# Set GPIO pins as output
GPIO.setup(LED_ONE_FINGER, GPIO.OUT)
GPIO.setup(LED_TWO_FINGERS, GPIO.OUT)
GPIO.setup(LED_THREE_FINGERS, GPIO.OUT)

# Initialize all LEDs to OFF
GPIO.output(LED_ONE_FINGER, GPIO.LOW)
GPIO.output(LED_TWO_FINGERS, GPIO.LOW)
GPIO.output(LED_THREE_FINGERS, GPIO.LOW)

# Global flag to track if any LED is currently active
led_active = False
# Global variable to track which object we're looking for next
next_detection = 1  # Start looking for 1 finger

# LED control function with callback when LED turns off
def turn_on_led(led_pin, duration=5):
    global led_active
    
    # If any LED is already active, don't proceed
    if led_active:
        return
    
    led_active = True
    
    def led_timer():
        global led_active, next_detection
        
        GPIO.output(led_pin, GPIO.HIGH)
        print(f"LED on GPIO {led_pin} is ON")
        time.sleep(duration)
        GPIO.output(led_pin, GPIO.LOW)
        print(f"LED on GPIO {led_pin} is OFF")
        
        # Update next detection target
        if led_pin == LED_ONE_FINGER:
            next_detection = 2
        elif led_pin == LED_TWO_FINGERS:
            next_detection = 3
        elif led_pin == LED_THREE_FINGERS:
            next_detection = 1
            
        led_active = False
    
    # Start the LED timer in a separate thread
    led_thread = threading.Thread(target=led_timer)
    led_thread.daemon = True
    led_thread.start()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set resolution to improve performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for detection validation
previous_finger_count = 0
stable_count = 0
STABILITY_THRESHOLD = 10  # Number of frames the same finger count must be detected for validation

# Improved function to detect fingers
def detect_fingers(frame):
    # Create a region of interest in the lower part of the frame
    # This helps avoid face detection as hands typically appear lower in the frame
    height, width = frame.shape[:2]
    roi_frame = frame[height//2:height, 0:width].copy()
    
    # Convert to HSV color space for better skin detection
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # Define range for skin color in HSV - adjusted for better accuracy
    lower_skin = np.array([0, 15, 60], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Draw ROI boundary for user feedback
    cv2.rectangle(frame, (0, height//2), (width, height), (0, 255, 255), 2)
    cv2.putText(frame, "Hand Detection Zone", (width//2-100, height//2-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    finger_count = 0
    confidence = 0
    
    if contours:
        # Find the largest contour by area
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        
        # Calculate the bounding rectangle of the contour for aspect ratio check
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Only proceed if area is large enough and has a reasonable aspect ratio for a hand
        if area > 5000 and aspect_ratio < 2.0 and aspect_ratio > 0.1:
            # Create a convex hull around the hand
            hull = cv2.convexHull(max_contour, returnPoints=False)
            
            # Find convexity defects
            defects = cv2.convexityDefects(max_contour, hull)
            
            # Draw contour and convex hull for visualization
            roi_display = roi_frame.copy()
            cv2.drawContours(roi_display, [max_contour], -1, (0, 255, 0), 2)
            
            defect_count = 0
            if defects is not None and len(defects) > 0:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])
                    
                    # Calculate distance between points
                    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    
                    # Calculate angle using cosine rule
                    if b*c == 0:
                        continue
                    
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / np.pi
                    
                    # If angle is less than 90 degrees, it's a finger
                    if angle <= 90 and d > 10000:  # Added depth threshold
                        defect_count += 1
                        # Draw defect point
                        cv2.circle(roi_display, far, 5, [0, 0, 255], -1)
            
            # Finger count is defects + 1
            finger_count = defect_count + 1
            
            # Limit finger count to 5 (max on a hand)
            finger_count = min(finger_count, 5)
            
            # Calculate confidence based on area and defect clarity
            confidence = min(area / 15000, 1.0) * 100
            
            # Show the contour image in the corner
            h_small, w_small = 120, 160
            roi_display_small = cv2.resize(roi_display, (w_small, h_small))
            frame[10:10+h_small, width-w_small-10:width-10] = roi_display_small
            
            # Draw a border around the small display
            cv2.rectangle(frame, (width-w_small-10, 10), (width-10, 10+h_small), (255, 255, 255), 1)
            cv2.putText(frame, "Hand Contour", (width-w_small-10, 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Display finger count and confidence on frame
    cv2.putText(frame, f"Detected fingers: {finger_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Only return finger count if confidence is high enough
    return finger_count if confidence > 50 else 0

try:
    print("Starting improved finger detection. Press Ctrl+C to exit.")
    print(f"Currently looking for {next_detection} finger(s)")
    
    # Add a short delay before starting detection to allow camera to warm up
    print("Warming up camera...")
    time.sleep(2)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip the frame horizontally (mirror)
        frame = cv2.flip(frame, 1)
        
        # Display which object/finger count we're looking for
        cv2.putText(frame, f"Looking for: {next_detection} finger(s)", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Only process detection if no LED is currently active
        if not led_active:
            # Detect fingers
            current_finger_count = detect_fingers(frame)
            
            # Implement temporal stability check
            if current_finger_count == previous_finger_count:
                stable_count += 1
            else:
                stable_count = 0
                previous_finger_count = current_finger_count
            
            # Display stability counter
            cv2.putText(frame, f"Stability: {stable_count}/{STABILITY_THRESHOLD}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            
            # Only trigger LED if finger count is stable for enough frames
            if stable_count >= STABILITY_THRESHOLD and current_finger_count == next_detection:
                # Reset stability counter
                stable_count = 0
                
                if next_detection == 1:
                    turn_on_led(LED_ONE_FINGER)
                    cv2.putText(frame, "One finger detected - LED 1 ON!", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif next_detection == 2:
                    turn_on_led(LED_TWO_FINGERS)
                    cv2.putText(frame, "Two fingers detected - LED 2 ON!", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif next_detection == 3:
                    turn_on_led(LED_THREE_FINGERS)
                    cv2.putText(frame, "Three fingers detected - LED 3 ON!", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Show message that system is waiting for current LED to turn off
            cv2.putText(frame, "Waiting for LED to turn off...", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Improved Finger Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user")
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    print("Program ended and GPIO cleaned up")
