import cv2
import numpy as np
import time
from picamera2 import Picamera2
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- 1. CONFIGURATION ---
SHOW_DISPLAY = True
CAMERA_RES = (320, 240)
# Updated to match your local template directory
REFERENCE_FOLDER = "templates/" 
MIN_MATCH_COUNT = 25

# --- 2. INITIALIZE ADVANCED SCANNERS ---
# ORB for Complex Symbols (QR, 3R, Biometric, Warning, Button)
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ArUco for the Grid Blocks
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Load Reference Images into Memory
reference_data = []
print(f"Loading reference symbols from {REFERENCE_FOLDER}...")
if os.path.exists(REFERENCE_FOLDER):
    for filename in os.listdir(REFERENCE_FOLDER):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(REFERENCE_FOLDER, filename)
            ref_img = cv2.imread(img_path, 0)
            if ref_img is None: continue
            
            kp, des = orb.detectAndCompute(ref_img, None)
            
            # ORB only stores templates that actually have complex features. 
            # Solid geometric shapes will naturally be skipped here.
            if des is not None and len(kp) > 0:
                # Format the name nicely (e.g., "qr" -> "QR", "warning" -> "Warning")
                raw_name = filename.split('.')[0]
                if raw_name.lower() == 'qr': clean_name = "QR"
                elif raw_name.lower() == '3r': clean_name = "3R"
                else: clean_name = raw_name.title()

                reference_data.append({
                    "name": clean_name, 
                    "descriptors": des
                })
                print(f"Loaded ORB Template: {clean_name}")
else:
    print(f"WARNING: Folder '{REFERENCE_FOLDER}' not found! ORB will be disabled.")

# --- 3. MATH FUNCTIONS ---
def are_lines_parallel(pt1, pt2, pt3, pt4, tolerance=15):
    dx1, dy1 = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dx2, dy2 = pt4[0] - pt3[0], pt4[1] - pt3[1]
    
    angle1 = np.degrees(np.arctan2(dy1, dx1))
    angle2 = np.degrees(np.arctan2(dy2, dx2))
    
    diff = abs(angle1 - angle2) % 180
    if diff > 90:
        diff = 180 - diff
    return diff < tolerance

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

# --- 4. MASTER VISION PIPELINE ---
def process_vision(roi_image):
    annotated_image = roi_image.copy() if SHOW_DISPLAY else None
    
    # Always create the saturation mask so we can view it in Window 1
    hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    _, mask = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # PRIORITY 1: ARUCO GRID MARKER
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        if SHOW_DISPLAY:
            cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
        return annotated_image, mask, f"GRID ID: {ids[0][0]}"

    # PRIORITY 2: ORB COMPLEX SYMBOLS
    kp_live, des_live = orb.detectAndCompute(gray, None)
    if des_live is not None:
        max_matches = 0
        best_label = None
        for ref in reference_data:
            matches = bf.match(ref["descriptors"], des_live)
            good_matches = [m for m in matches if m.distance < 50]
            if len(good_matches) > max_matches and len(good_matches) > MIN_MATCH_COUNT:
                max_matches = len(good_matches)
                best_label = ref["name"]
        
        if best_label:
            print(f"Target Acquired: {best_label} (Matches: {max_matches})")
            return annotated_image, mask, best_label

    # PRIORITY 3: GEOMETRIC MATH LOGIC
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_shape_name = "NONE"

    for c in contours:
        area = cv2.contourArea(c)
        if area < 300: 
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        vertices = len(approx)
        
        hull = cv2.convexHull(c)
        solidity = area / float(cv2.contourArea(hull)) if cv2.contourArea(hull) > 0 else 0

        shape_name = "Unknown"

        # YOUR DIALED-IN DECISION TREE (Updated names)
        if 11 <= vertices <= 14 and solidity < 0.90:
            shape_name = "Plus" 
        elif vertices == 10 and solidity < 0.6:
            shape_name = "Star"
        elif vertices == 8 and solidity > 0.85:
            shape_name = "Octagon"
        elif 6 <= vertices <= 9 and 0.4 < solidity < 0.75:
            direction = get_arrow_direction(mask, x, y, w, h)
            shape_name = f"{direction} Arrow"
        elif vertices == 4:
            pts = approx.reshape(4, 2)
            pair1 = are_lines_parallel(pts[0], pts[1], pts[2], pts[3])
            pair2 = are_lines_parallel(pts[1], pts[2], pts[3], pts[0])
            shape_name = "Trapezium" if pair1 ^ pair2 else "Diamond"
        else:
            if solidity > 0.92:
                shape_name = "Segment"
            elif solidity > 0.75 and solidity <= 0.92:
                shape_name = "Packman"

        best_shape_name = shape_name
        print(f"Target Acquired: {shape_name} (V:{vertices}, S:{solidity:.2f})")

        if SHOW_DISPLAY:
            cv2.drawContours(annotated_image, [c], -1, (0, 255, 0), 2)
            cv2.putText(annotated_image, shape_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return annotated_image, mask, best_shape_name


# --- 5. MAIN LOOP ---
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": CAMERA_RES, "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

print("\n--- ULTIMATE VISION SYSTEM ACTIVE ---")

try:
    while True:
        frame = picam2.capture_array()
        
        # Crop to the floor (bottom 220px frame based on 320x240 res)
        floor_roi = frame[20:240, 0:320] 
        
        result_view, debug_mask, status_label = process_vision(floor_roi)

        if SHOW_DISPLAY and result_view is not None:
            # Add the overall status to the top corner of the screen
            color = (0, 255, 0) if status_label != "NONE" else (0, 0, 255)
            cv2.putText(result_view, f"STATUS: {status_label}", (5, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show the Windows
            cv2.imshow("1. The Math Mask", debug_mask)
            cv2.imshow("2. Labeled Targets", result_view)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
