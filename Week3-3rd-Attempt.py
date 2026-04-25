import cv2
import numpy as np
import time
import multiprocessing
import os
import RPi.GPIO as GPIO
from picamera2 import Picamera2

os.environ["QT_QPA_PLATFORM"] = "xcb"

# ======================
# 1. CONFIGURATION
# ======================
SHOW_DISPLAY = True       # Set False during race
CAMERA_RES   = (320, 240)
REFERENCE_FOLDER = "templates/"

MIN_MATCH_COUNT  = 15

SYMBOL_ACTIONS = {
    "Button":      "STOP",
    "Warning":     "HALT_3_SEC",
    "3R":          "TURN_RIGHT_360",
    "Left Arrow":  "TURN_LEFT_90",
    "Right Arrow": "TURN_RIGHT_90",
    "Biometric":   "DISPLAY",
    "QR":          "DISPLAY",
    "Up Arrow":    "FORWARD"
}

# ======================
# 2. VISION MATH HELPERS
# ======================
def are_lines_parallel(pt1, pt2, pt3, pt4, tolerance=15):
    dx1, dy1 = pt2[0]-pt1[0], pt2[1]-pt1[1]
    dx2, dy2 = pt4[0]-pt3[0], pt4[1]-pt3[1]
    angle1 = np.degrees(np.arctan2(dy1, dx1))
    angle2 = np.degrees(np.arctan2(dy2, dx2))
    diff = abs(angle1 - angle2) % 180
    if diff > 90: diff = 180 - diff
    return diff < tolerance

def get_arrow_direction(mask, x, y, w, h):
    arrow_box = mask[y:y+h, x:x+w]
    M = cv2.moments(arrow_box)
    if M["m00"] == 0: return "Unknown"
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    dx = cx - (w // 2)
    dy = cy - (h // 2)
    if abs(dx) > abs(dy): return "Right" if dx > 0 else "Left"
    else: return "Down" if dy > 0 else "Up"

# ======================
# 3. COLOR DETECTION HELPER
# ======================
def get_color_masks(rgb_roi):
    blur = cv2.GaussianBlur(rgb_roi, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    mask_red = cv2.inRange(hsv, (105, 120, 60), (145, 255, 255))
    mask_yellow = cv2.inRange(hsv, (75, 120, 60), (115, 255, 255))

    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    _, mask_black = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    color_any  = cv2.bitwise_or(mask_red, mask_yellow)
    mask_black = cv2.bitwise_and(mask_black, cv2.bitwise_not(color_any))

    return mask_red, mask_yellow, mask_black

# ======================
# 4. AI VISION WORKER
# ======================
def ai_vision_worker(image_queue, command_queue):
    print("[AI Worker] Booting models on Core 2...")

    orb = cv2.ORB_create(nfeatures=1000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    reference_data = []
    if os.path.exists(REFERENCE_FOLDER):
        for filename in os.listdir(REFERENCE_FOLDER):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(REFERENCE_FOLDER, filename)
                ref_img  = cv2.imread(img_path, 0)
                if ref_img is None: continue
                kp, des = orb.detectAndCompute(ref_img, None)
                if des is not None and len(kp) > 0:
                    raw_name   = filename.split('.')[0]
                    if raw_name.lower() == 'qr':  clean_name = "QR"
                    elif raw_name.lower() == '3r': clean_name = "3R"
                    else:                          clean_name = raw_name.title()
                    reference_data.append({"name": clean_name, "descriptors": des})

    print("[AI Worker] Ready.")

    while True:
        if not image_queue.empty():
            ai_roi = image_queue.get()
            gray   = cv2.cvtColor(ai_roi, cv2.COLOR_BGR2GRAY)
            best_shape_name = "Unknown"

            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                best_shape_name = f"GRID ID: {ids[0][0]}"
            else:
                kp_live, des_live = orb.detectAndCompute(gray, None)
                if des_live is not None:
                    max_matches = 0
                    for ref in reference_data:
                        matches      = bf.match(ref["descriptors"], des_live)
                        good_matches = [m for m in matches if m.distance < 50]
                        if len(good_matches) > max_matches and len(good_matches) > MIN_MATCH_COUNT:
                            max_matches     = len(good_matches)
                            best_shape_name = ref["name"]

                if best_shape_name == "Unknown":
                    _, mask     = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    img_h, img_w = gray.shape

                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < 300 or area > 8000: continue
                        x, y, w, h = cv2.boundingRect(c)
                        if x < 5 or y < 5 or (x+w) > (img_w-5) or (y+h) > (img_h-5): continue
                        if h == 0: continue

                        aspect_ratio = float(w) / float(h)
                        peri     = cv2.arcLength(c, True)
                        
                        # Strict approximation for complex shapes
                        approx   = cv2.approxPolyDP(c, 0.015 * peri, True)
                        vertices = len(approx)
                        
                        # Loose approximation for Triangles
                        approx_loose   = cv2.approxPolyDP(c, 0.04 * peri, True)
                        loose_vertices = len(approx_loose)
                        
                        hull     = cv2.convexHull(c)
                        solidity = area / float(cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else 0

                        if 11 <= vertices <= 14 and solidity < 0.90:
                            best_shape_name = "Plus"
                        elif vertices == 10 and solidity < 0.6:
                            best_shape_name = "Star"
                        elif vertices == 8 and solidity > 0.85:
                            best_shape_name = "Octagon"
                        elif 6 <= vertices <= 9 and 0.35 < solidity < 0.85 and 0.3 <= aspect_ratio <= 2.5:
                            best_shape_name = f"{get_arrow_direction(mask, x, y, w, h)} Arrow"
                        elif vertices == 4:
                            pts   = approx.reshape(4, 2)
                            pair1 = are_lines_parallel(pts[0], pts[1], pts[2], pts[3])
                            pair2 = are_lines_parallel(pts[1], pts[2], pts[3], pts[0])
                            best_shape_name = "Trapezium" if pair1 ^ pair2 else "Diamond"
                        
                        # The Triangle/Warning Sign Fix
                        elif loose_vertices == 3 and solidity > 0.75:
                            best_shape_name = "Warning"
                            
                        else:
                            if solidity > 0.92:           best_shape_name = "Segment"
                            elif 0.75 < solidity <= 0.92: best_shape_name = "Packman"

            if best_shape_name in SYMBOL_ACTIONS:
                action = SYMBOL_ACTIONS[best_shape_name]
                print(f"[AI Worker] Detected '{best_shape_name}'. Sending command: {action}")
                while not command_queue.empty(): command_queue.get()
                command_queue.put(action)

# ======================
# 5. MAIN DRIVER
# ======================
def main():
    image_queue   = multiprocessing.Queue(maxsize=1)
    command_queue = multiprocessing.Queue(maxsize=1)

    ai_process = multiprocessing.Process(target=ai_vision_worker, args=(image_queue, command_queue))
    ai_process.start()

    # --- MOTOR SETUP ---
    GPIO.setwarnings(False) 
    
    try:
        GPIO.cleanup()
    except:
        pass

    GPIO.setmode(GPIO.BCM)

    ENA, IN1, IN2, ENB, IN3, IN4 = 17, 27, 22, 5, 6, 13
    
    for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
        GPIO.setup(pin, GPIO.OUT)
        
    pwmA = GPIO.PWM(ENA, 300)
    pwmB = GPIO.PWM(ENB, 300)
    pwmA.start(0)
    pwmB.start(0)

    def set_motor(left, right):
        GPIO.output(IN1, GPIO.LOW  if left  >= 0 else GPIO.HIGH)
        GPIO.output(IN2, GPIO.HIGH if left  >= 0 else GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW  if right >= 0 else GPIO.HIGH)
        GPIO.output(IN4, GPIO.HIGH if right >= 0 else GPIO.LOW)
        pwmA.ChangeDutyCycle(min(abs(left),  100))
        pwmB.ChangeDutyCycle(min(abs(right), 100))

    # --- CAMERA SETUP ---
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": CAMERA_RES, "format": "RGB888"}))
    picam2.start()
    time.sleep(2)
    
    picam2.set_controls({"ExposureTime": 15000, "AnalogueGain": 2.0})

    WIDTH  = CAMERA_RES[0]
    CENTER = WIDTH // 2

    # --- PID ---
    KP = 0.015 
    KI = 0.0001
    KD = 0.002
    
    MAX_SPEED = 20

    prev_error = 0
    integral   = 0
    last_error = 0
    prev_time  = time.time()

    action_cooldowns  = {}
    COOLDOWN_DURATION = 7.0

    prev_tracked_color = "BLACK"
    curve_accumulator = 0
    HEADING_LOCK_DURATION = 2.5

    print("--- ROBOT IS ACTIVE ---")

    try:
        while True:
            frame        = picam2.capture_array()
            current_time = time.time()

            # -----------------------------------------------
            # FSM COMMAND EXECUTION
            # -----------------------------------------------
            if not command_queue.empty():
                action = command_queue.get()

                if action in action_cooldowns and \
                   (current_time - action_cooldowns[action]) < COOLDOWN_DURATION:
                    pass
                else:
                    action_cooldowns[action] = current_time
                    print(f"[Main OS] EXECUTING OVERRIDE: {action}")

                    if action == "HALT_3_SEC":
                        set_motor(0, 0)
                        time.sleep(3)

                    elif action == "TURN_LEFT_90":
                        set_motor(50, -50)
                        time.sleep(0.9)

                    elif action == "TURN_RIGHT_90":
                        set_motor(-50, 50)
                        time.sleep(0.9)

                    elif action == "TURN_RIGHT_360":
                        set_motor(-50, 50)
                        time.sleep(3.6)

                    elif action == "STOP":
                        set_motor(0, 0)
                        time.sleep(5)

                    elif action == "DISPLAY":
                        print("Displaying Info... Continuing with Line Following...")

                    elif action == "FORWARD":
                        set_motor(40,40)
                        time.sleep(1.5)

                    if action != "DISPLAY":
                        while not command_queue.empty(): command_queue.get()
                        prev_time = time.time()
                        continue

            # -----------------------------------------------
            # FEED AI WORKER
            # -----------------------------------------------
            if image_queue.empty():
                ai_crop = frame[0:180, :]
                image_queue.put(ai_crop)

            # -----------------------------------------------
            # COLOR-PRIORITY LINE DETECTION
            # -----------------------------------------------
            roi_lower = frame[120:240, :]
            roi_upper = frame[80:120,  :]

            mr_lo, my_lo, mb_lo = get_color_masks(roi_lower)
            mr_up, my_up, _     = get_color_masks(roi_upper)

            mask_red_combined    = np.vstack((mr_up, mr_lo))
            mask_yellow_combined = np.vstack((my_up, my_lo))

            # ---------------------------------------------------------
            # PERIPHERAL VISION TRIPWIRES
            # ---------------------------------------------------------
            left_red_pixels = cv2.countNonZero(mask_red_combined[:, 0:40])
            right_red_pixels = cv2.countNonZero(mask_red_combined[:, 280:320])
            left_yel_pixels = cv2.countNonZero(mask_yellow_combined[:, 0:40])
            right_yel_pixels = cv2.countNonZero(mask_yellow_combined[:, 280:320])

            TRIPWIRE_THRESHOLD = 150
            tripwire_triggered = False

            if prev_tracked_color == "BLACK":
                if left_red_pixels > TRIPWIRE_THRESHOLD or left_yel_pixels > TRIPWIRE_THRESHOLD:
                    print("[TRIPWIRE] Acute line detected in LEFT peripheral vision! Snapping left...")
                    set_motor(65, -65) 
                    time.sleep(0.2)
                    tripwire_triggered = True
                
                elif right_red_pixels > TRIPWIRE_THRESHOLD or right_yel_pixels > TRIPWIRE_THRESHOLD:
                    print("[TRIPWIRE] Acute line detected in RIGHT peripheral vision! Snapping right...")
                    set_motor(-65, 65) 
                    time.sleep(0.2)
                    tripwire_triggered = True

            if tripwire_triggered:
                integral = 0
                prev_time = current_time
                continue
            # ---------------------------------------------------------

            c_red, _ = cv2.findContours(mask_red_combined,    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_yel, _ = cv2.findContours(mask_yellow_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c_blk, _ = cv2.findContours(mb_lo,                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ---------------------------------------------------------
            # FILTER BY AREA AND WIDTH (AND DETECT WARNING SIGN!)
            # ---------------------------------------------------------
            MAX_LINE_WIDTH = 90  

            valid_red = []
            valid_yel = []
            valid_blk = [c for c in c_blk if cv2.contourArea(c) > 200]

            for c in c_red:
                if cv2.contourArea(c) > 80 and cv2.boundingRect(c)[2] < MAX_LINE_WIDTH:
                    valid_red.append(c)

            for c in c_yel:
                area = cv2.contourArea(c)
                if area > 80:
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # EXTENT MATH: Area divided by Bounding Box Area
                    # Diagonal lines = Low Extent (~0.4). Solid Signs = High Extent (~0.9)
                    extent = area / float(w * h) if (w * h) > 0 else 0

                    if w < MAX_LINE_WIDTH or extent < 0.6:
                        # If it is narrow OR it is a diagonal stripe, it's a driving line
                        valid_yel.append(c)
                    elif w >= MAX_LINE_WIDTH and area > 4000 and extent >= 0.6:
                        # It is too fat, massive, AND solid like a block. This is the sign!
                        if command_queue.empty():
                            print(f"[Color Vision] Massive Yellow Sign detected (Width: {w}, Extent: {extent:.2f}). Triggering HALT!")
                            command_queue.put("HALT_3_SEC")

            contours     = None
            active_color = "BLACK"

            if valid_red:
                contours     = valid_red
                active_color = "RED"
            elif valid_yel:
                contours     = valid_yel
                active_color = "YELLOW"
            elif valid_blk:
                contours     = valid_blk
                active_color = "BLACK"

            # -----------------------------------------------
            # THE BLIND ARC MERGE HANDLER
            # -----------------------------------------------
            if prev_tracked_color in ("RED", "YELLOW") and active_color == "BLACK":
                print(f"[MERGE] Shortcut ended. Curve Momentum: {curve_accumulator}")
                
                if curve_accumulator > 10:
                    print("[MERGE] Forcing LEFT arc to bypass loop!")
                    set_motor(20, 60) 
                    time.sleep(0.35)
                
                elif curve_accumulator < -10:
                    print("[MERGE] Forcing RIGHT arc to bypass loop!")
                    set_motor(60, 20) 
                    time.sleep(0.35)
                
                curve_accumulator = 0
                prev_tracked_color = "BLACK"
                integral = 0
                prev_time = time.time()
                continue 

            # -----------------------------------------------
            # DRIVE
            # -----------------------------------------------
            if contours:
                c = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(c)

                if active_color in ("RED", "YELLOW") and contour_area > 80:
                    valid_to_drive = True
                elif active_color == "BLACK" and contour_area > 200:
                    valid_to_drive = True
                else:
                    valid_to_drive = False

                if valid_to_drive:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx    = int(M["m10"] / M["m00"])
                        error = cx - CENTER

                        if active_color != prev_tracked_color:
                            integral           = 0
                            prev_error         = error
                            prev_tracked_color = active_color
                            print(f"[TRACKING] Transitioned to {active_color} line")

                        if active_color in ("RED", "YELLOW"):
                            if error > 5:
                                curve_accumulator += 1
                            elif error < -5:
                                curve_accumulator -= 1

                        if abs(error) < 8:
                            error = 0

                        abs_error  = abs(error)
                        last_error = error

                        DYNAMIC_MAX_SPEED = 35 if active_color in ("RED", "YELLOW") \
                                               or contour_area > 3500 \
                                            else MAX_SPEED

                        if abs_error > 110:
                            if error > 0: set_motor(-60, 60)
                            else:         set_motor(60, -60)
                            prev_time = current_time
                            continue

                        if abs_error > 70:
                            if error > 0: set_motor(-25, 65)
                            else:         set_motor(65, -25)
                            prev_time = current_time
                            continue

                        dt = max(current_time - prev_time, 0.001)

                        integral  += error * dt
                        integral   = max(-500, min(500, integral))
                        derivative = (error - prev_error) / dt

                        correction = (KP * error) + (KI * integral) + (KD * derivative)
                        correction = max(-80, min(80, correction))

                        prev_error = error
                        prev_time  = current_time

                        left  = DYNAMIC_MAX_SPEED - correction
                        right = DYNAMIC_MAX_SPEED + correction
                        set_motor(left, right)

            else:
                set_motor(-55, 55) if last_error > 0 else set_motor(55, -55)
                prev_time = current_time

            # -----------------------------------------------
            # VISUAL DEBUG
            # -----------------------------------------------
            if SHOW_DISPLAY:
                display_frame = frame.copy()
                
                cv2.rectangle(display_frame, (0, 10), (40, 240), (0, 0, 255), 1)
                cv2.rectangle(display_frame, (280, 10), (320, 240), (0, 0, 255), 1)
                
                cv2.rectangle(display_frame, (0, 120), (320, 240), (0, 255, 0),   2)
                cv2.rectangle(display_frame, (0, 80),  (320, 120),  (255, 165, 0), 1)

                cv2.putText(display_frame, f"Tracking: {active_color}", (5, 235),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                cv2.imshow("Camera",       display_frame)
                cv2.imshow("Red Mask",     mask_red_combined)
                cv2.imshow("Yellow Mask",  mask_yellow_combined)
                cv2.imshow("Black Mask",   mb_lo)

                if cv2.waitKey(1) == ord('q'):
                    break

    except KeyboardInterrupt:
        pass

    finally:
        pwmA.stop()
        pwmB.stop()
        try:
            GPIO.cleanup()
        except:
            pass
        picam2.stop()
        cv2.destroyAllWindows()
        ai_process.terminate()
        ai_process.join()

if __name__ == '__main__':
    main()
