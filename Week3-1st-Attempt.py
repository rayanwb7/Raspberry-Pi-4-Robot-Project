import cv2
import numpy as np
import time
import multiprocessing
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2

os.environ["QT_QPA_PLATFORM"] = "xcb"


# --- 1. CONFIGURATION & COMMAND MAPPING ---
SHOW_DISPLAY = True  # <--- TOGGLE THIS TO FALSE FOR FINAL RACING
CAMERA_RES = (320, 240)
REFERENCE_FOLDER = "templates/" 
MIN_MATCH_COUNT = 25

# Map your AI detection labels to strict FSM Commands
SYMBOL_ACTIONS = {
    "Button": "STOP",
    "Warning": "HALT_3_SEC",
    "3R": "TURN_RIGHT_360",
    "Left Arrow": "TURN_LEFT_90",
    "Right Arrow": "TURN_RIGHT_90",
    "Biometric":"DISPLAY",
    "QR":"DISPLAY",
    "Up Arrow": "FORWARD"
}

# --- 2. VISION MATH HELPERS ---
def are_lines_parallel(pt1, pt2, pt3, pt4, tolerance=15):
    dx1, dy1 = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dx2, dy2 = pt4[0] - pt3[0], pt4[1] - pt3[1]
    angle1 = np.degrees(np.arctan2(dy1, dx1))
    angle2 = np.degrees(np.arctan2(dy2, dx2))
    diff = abs(angle1 - angle2) % 180
    if diff > 90: diff = 180 - diff
    return diff < tolerance

#calculating the pixels of the arrows
def get_arrow_direction(mask, x, y, w, h):
    arrow_box = mask[y:y+h, x:x+w]
    top = cv2.countNonZero(arrow_box[0:h//2, 0:w])
    bot = cv2.countNonZero(arrow_box[h//2:h, 0:w])
    lft = cv2.countNonZero(arrow_box[0:h, 0:w//2])
    rgt = cv2.countNonZero(arrow_box[0:h, w//2:w])
    if abs(top - bot) > abs(lft - rgt):
        return "Up" if top > bot else "Down"
    else:
        return "Left" if lft > rgt else "Right"

# --- 3. AI VISION WORKER (RUNS ON CORE 2) ---
def ai_vision_worker(image_queue, command_queue):
    """Isolated process. Analyzes frames for symbols and outputs FSM commands."""
    print("[AI Worker] Booting models on Core 2...")
    
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    reference_data = []
    if os.path.exists(REFERENCE_FOLDER):
        for filename in os.listdir(REFERENCE_FOLDER):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(REFERENCE_FOLDER, filename)
                ref_img = cv2.imread(img_path, 0)
                if ref_img is None: continue
                kp, des = orb.detectAndCompute(ref_img, None)
                if des is not None and len(kp) > 0:
                    raw_name = filename.split('.')[0]
                    if raw_name.lower() == 'qr': 
                        clean_name = "QR"
                    elif raw_name.lower() == '3r': 
                        clean_name = "3R"
                    else: 
                        clean_name = raw_name.title()
                    reference_data.append({"name": clean_name, "descriptors": des})
    
    print("[AI Worker] Ready.")

    while True:
        if not image_queue.empty():
            frame = image_queue.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            best_shape_name = "Unknown"

            # 1. ArUco
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                best_shape_name = f"GRID ID: {ids[0][0]}"
            else:
                # 2. ORB Matches
                kp_live, des_live = orb.detectAndCompute(gray, None)
                if des_live is not None:
                    max_matches = 0
                    for ref in reference_data:
                        matches = bf.match(ref["descriptors"], des_live)
                        good_matches = [m for m in matches if m.distance < 50]
                        if len(good_matches) > max_matches and len(good_matches) > MIN_MATCH_COUNT:
                            max_matches = len(good_matches)
                            best_shape_name = ref["name"]
                
                # 3. Geometric Math (Fallback)
                if best_shape_name == "Unknown":
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    _, s, _ = cv2.split(hsv)
                    _, mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for c in contours:
                        if cv2.contourArea(c) < 300: continue
                        x, y, w, h = cv2.boundingRect(c)
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
                        vertices = len(approx)
                        
                        hull = cv2.convexHull(c)
                        solidity = cv2.contourArea(c) / float(cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else 0
                        
                        if 11 <= vertices <= 14 and solidity < 0.90: best_shape_name = "Plus" 
                        elif vertices == 10 and solidity < 0.6: best_shape_name = "Star"
                        elif vertices == 8 and solidity > 0.85: best_shape_name = "Octagon"
                        elif 6 <= vertices <= 9 and 0.4 < solidity < 0.75:
                            best_shape_name = f"{get_arrow_direction(mask, x, y, w, h)} Arrow"
                        elif vertices == 4:
                            pts = approx.reshape(4, 2)
                            pair1 = are_lines_parallel(pts[0], pts[1], pts[2], pts[3])
                            pair2 = are_lines_parallel(pts[1], pts[2], pts[3], pts[0])
                            best_shape_name = "Trapezium" if pair1 ^ pair2 else "Diamond"
                        else:
                            # If it's curved (many vertices) we use solidity to guess
                            if solidity > 0.92:
                                best_shape_name = "Segment"
                            elif 0.75 < solidity <= 0.92:
                                best_shape_name = "Packman"
            #To see what OpenCV is mathematically thinking!
                        if best_shape_name == "Unknown":
                            print(f"[AI Debug] Saw an unknown shape -> Vertices: {vertices}, Solidity: {solidity:.2f}")
            if best_shape_name in SYMBOL_ACTIONS:
                action = SYMBOL_ACTIONS[best_shape_name]
                print(f"[AI Worker] Detected '{best_shape_name}'. Sending command: {action}")
                
                while not command_queue.empty(): command_queue.get()
                command_queue.put(action)

# --- 4. MAIN DRIVER (RUNS ON CORE 1) ---
def main():
    image_queue = multiprocessing.Queue(maxsize=1)
    command_queue = multiprocessing.Queue(maxsize=1)
    
    ai_process = multiprocessing.Process(
        target=ai_vision_worker, 
        args=(image_queue, command_queue)
    )
    ai_process.start()

    GPIO.setmode(GPIO.BCM)
    ENA, IN1, IN2, ENB, IN3, IN4 = 17, 27, 22, 5, 6, 13
    GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)
    pwmA = GPIO.PWM(ENA, 1000)
    pwmB = GPIO.PWM(ENB, 1000)
    pwmA.start(0)
    pwmB.start(0)

    def set_motor(left, right):
        GPIO.output(IN1, GPIO.LOW if left >= 0 else GPIO.HIGH)
        GPIO.output(IN2, GPIO.HIGH if left >= 0 else GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW if right >= 0 else GPIO.HIGH)
        GPIO.output(IN4, GPIO.HIGH if right >= 0 else GPIO.LOW)
        pwmA.ChangeDutyCycle(min(abs(left), 100))
        pwmB.ChangeDutyCycle(min(abs(right), 100))

    KP, KI, KD = 0.45, 0.0004, 0.14
    MAX_SPEED = 40
    CENTER = 160 
    prev_error, integral, last_error = 0, 0, 0
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": CAMERA_RES, "format": "RGB888"}))
    picam2.start()
    time.sleep(2)

    print("--- ROBOT IS ACTIVE ---")
    prev_time = time.perf_counter()

    try:
        while True:
            frame = picam2.capture_array()
            
            # --- 5. EXECUTE AI COMMANDS (FSM OVERRIDE) ---
            if not command_queue.empty():
                action = command_queue.get()
                print(f"[Main OS] EXECUTING OVERRIDE: {action}")
                
                if action == "HALT_3_SEC":
                    set_motor(0, 0)
                    time.sleep(3)
                elif action == "TURN_LEFT_90":
                    set_motor(-40, 40)
                    time.sleep(1.5) 
                elif action == "TURN_RIGHT_90":
                    set_motor(40, -40)
                    time.sleep(1.5)
                elif action =="TURN_RIGHT_360":
                    set_motor(40,-40)
                    time.sleep(3)
                elif action=="STOP":
                    set_motor(0,0)
                    time.sleep(5)
                elif action=="FORWARD":
                    set_motor(40,40)
                    time.sleep(2)
                elif action=="DISPLAYING":
                    print("Continuing with Line Following...")
                
                #clear queue
                while not command_queue.empty(): command_queue.get()
                prev_time = time.perf_counter()
                continue

            # --- 6. FEED AI WORKER ---
            if image_queue.empty():
                image_queue.put(frame)

            # --- 7. OPTIMIZED LINE FOLLOWING ---
            roi_frame = frame[130:200, :] 
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_time = time.perf_counter()
            dt = max(current_time - prev_time, 0.001)

            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 200:
                    M = cv2.moments(c)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else CENTER
                    
                    error = cx - CENTER
                    last_error = error

                    if abs(error) > 120:
                        set_motor(-100, 100) if error > 0 else set_motor(100, -100)
                        prev_time = current_time
                    elif abs(error) > 70:
                        set_motor(-70, 90) if error > 0 else set_motor(90, -70)
                        prev_time = current_time
                    else:
                        integral = max(-500, min(500, integral + (error * dt)))
                        derivative = (error - prev_error) / dt
                        correction = max(-120, min(120, (KP * error) + (KI * integral) + (KD * derivative)))

                        prev_error = error
                        prev_time = current_time

                        set_motor(MAX_SPEED - correction, MAX_SPEED + correction)
                else:
                    prev_time = current_time
            else:
                set_motor(-80, 80) if last_error > 0 else set_motor(80, -80)
                prev_time = current_time

            # --- 8. VISUAL DEBUGGING DISPLAY ---
            if SHOW_DISPLAY:
                # Draw a green rectangle showing where the line-follower is looking
                display_frame = frame.copy()
                cv2.rectangle(display_frame, (0, 130), (320, 200), (0, 255, 0), 2)
                
                # Show the windows
                cv2.imshow("Main Camera (Green Box = Line ROI)", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                cv2.imshow("Line Follower Mask", thresh)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit signal received...")
                    break

    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        pwmA.stop()
        pwmB.stop()
        GPIO.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()
        ai_process.terminate()
        ai_process.join()

if __name__ == '__main__':
    main()
