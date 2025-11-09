import cv2
import mediapipe as mp
import numpy as np
import requests
import base64
import json
import time
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading

# ===== FLASK SETUP (MUST BE FIRST) =====
app = Flask(__name__, static_folder='.')
CORS(app)

app_state = {
    'drawing_active': True,
    'analyzing': False,
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/status')
def get_status():
    return jsonify(app_state)

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

print("\n" + "="*60)
print("🚀 FLASK SERVER STARTED!")
print("📱 Open this in your browser: http://localhost:5000")
print("="*60 + "\n")
time.sleep(2)  # Give Flask time to start

# ===== YOUR ORIGINAL CODE BELOW =====

CANVAS_SIZE = 500
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 8
DOT_COLOR = (0, 0, 255)
SMOOTHING_FACTOR = 0.3
smoothed_point = None

apiKey = "AIzaSyCUiHq0ZnFOIHuVSh0SSvC_0jVkJ0vOUL8"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

THUMB_TIP_ID = 4
THUMB_BASE_ID = 2
INDEX_FINGER_TIP_ID = 8
FINGER_TIPS = [8, 12, 16, 20]
BASE_KNUCKLES = [5, 9, 13, 17]
PIP_JOINTS = [6, 10, 14, 18]

drawing_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
last_point = None
drawing_active = True
cooldown_counter = 0
COOLDOWN_FRAMES = 30
current_expression = "Draw a math problem (e.g., solve for x: 2x+3=7)"
calculation_result = None

def is_thumb_extended(landmarks):
    wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    thumb_tip_x = landmarks.landmark[THUMB_TIP_ID].x
    thumb_tip_y = landmarks.landmark[THUMB_TIP_ID].y
    thumb_base_x = landmarks.landmark[THUMB_BASE_ID].x
    thumb_base_y = landmarks.landmark[THUMB_BASE_ID].y
    
    dist_tip_wrist = np.linalg.norm(np.array([thumb_tip_x, thumb_tip_y]) - np.array([wrist_x, wrist_y]))
    dist_base_wrist = np.linalg.norm(np.array([thumb_base_x, thumb_base_y]) - np.array([wrist_x, wrist_y]))
    
    return (dist_tip_wrist > dist_base_wrist + 0.05)

def is_calculate_gesture(hand_landmarks):
    thumb_extended = is_thumb_extended(hand_landmarks)
    if not thumb_extended:
        return False
    
    all_fingers_curled = True
    for tip_id, base_id in zip(FINGER_TIPS, BASE_KNUCKLES):
        tip_y = hand_landmarks.landmark[tip_id].y
        base_y = hand_landmarks.landmark[base_id].y
        if tip_y < base_y:
            all_fingers_curled = False
            break
    
    return thumb_extended and all_fingers_curled

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def call_gemini_api(base64_image):
    time.sleep(0.5)
    
    system_prompt = (
        "You are an expert mathematical solver and interpreter. Analyze the provided handwritten image. "
        "Your task is to identify the mathematical problem (which may include algebra, calculus, or graphing equations), "
        "cleanly transcribe it, and provide the step-by-step solution and final answer. "
        "Format the solution using Markdown, including LaTeX/math syntax for clarity. "
        "Do not use weird symbols like $$ and other LaTeX math symbols just in pure raw form and explanation should be elaborate for hard problems."
        "If you cannot confidently read a valid math problem, return 'INVALID' for all fields."
    )
    
    prompt = "Read the mathematical problem from this image and provide the result in the specified JSON format."
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "problem": {"type": "STRING"},
                    "solution_type": {"type": "STRING"},
                    "solution_text": {"type": "STRING"}
                }
            }
        }
    }

    try:
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
                response.raise_for_status()
                
                result_json = response.json()
                json_string = result_json['candidates'][0]['content']['parts'][0]['text']
                parsed_data = json.loads(json_string)
                
                return parsed_data
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise e
    except requests.exceptions.HTTPError as err:
        return {"problem": f"API Error: {err.response.status_code}", "solution_text": "Could not connect to the math solver.", "solution_type": "Error"}
    except Exception as e:
        return {"problem": f"API/Parsing Error: {e}", "solution_text": "There was an issue processing the response.", "solution_type": "Error"}

cap = cv2.VideoCapture(0)
cv2.namedWindow('Gesture-Controlled Calculator')

print("--- AI Handwriting Symbolic Math Solver ---")
print("Instructions:")
print("1. **Cursor Control:** Use your extended Index finger to move the cursor.")
print("2. **Draw Toggle (Start/Stop):** Press the **'s' key** to toggle drawing ON/OFF.")
print("3. **Clear Canvas:** Press the **'c' key**.")
print("4. **SOLVE (Trigger API):** Hold up ONLY your Thumb.")
print("5. Press 'q' to quit.")
print("---------------------------------")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    current_point = None
    gesture_command = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        raw_x = int(hand_landmarks.landmark[INDEX_FINGER_TIP_ID].x * w)
        raw_y = int(hand_landmarks.landmark[INDEX_FINGER_TIP_ID].y * h)
        raw_finger_point = (raw_x, raw_y)

        if smoothed_point is None:
            smoothed_point = raw_finger_point
        else:
            smooth_x, smooth_y = smoothed_point
            new_smooth_x = int(smooth_x * (1 - SMOOTHING_FACTOR) + raw_x * SMOOTHING_FACTOR)
            new_smooth_y = int(smooth_y * (1 - SMOOTHING_FACTOR) + raw_y * SMOOTHING_FACTOR)
            smoothed_point = (new_smooth_x, new_smooth_y)
        
        current_point = smoothed_point
        
        if is_calculate_gesture(hand_landmarks):
            gesture_command = 'CALCULATE'
        
        cv2.circle(image, current_point, 8, DOT_COLOR, -1)
    else:
        last_point = None
        smoothed_point = None
    
    if drawing_active and current_point:
        if last_point:
            cv_x1 = int(last_point[0] * (CANVAS_SIZE / w))
            cv_y1 = int(last_point[1] * (CANVAS_SIZE / h))
            cv_x2 = int(current_point[0] * (CANVAS_SIZE / w))
            cv_y2 = int(current_point[1] * (CANVAS_SIZE / h))
            cv2.line(drawing_canvas, (cv_x1, cv_y1), (cv_x2, cv_y2), LINE_COLOR, LINE_THICKNESS)
        last_point = current_point
    else:
        last_point = None
    
    if cooldown_counter > 0:
        cooldown_counter -= 1
        gesture_command = None

    if gesture_command == 'CALCULATE' and cooldown_counter == 0:
        print("--- SOLVE Command Received ---")
        current_expression = "CAPTURING & ANALYZING SYMBOLIC MATH..."
        calculation_result = None
        cooldown_counter = COOLDOWN_FRAMES
        
        # UPDATE UI STATE
        app_state['analyzing'] = True
        app_state['problem'] = current_expression
        
        temp_img = drawing_canvas.copy()
        base64_img = image_to_base64(temp_img)
        api_response = call_gemini_api(base64_img)
        
        current_expression = api_response.get("problem", "Could not read problem.")
        if api_response.get("problem") != "INVALID":
            result_type = api_response.get("solution_type", "Result")
            result_text = api_response.get("solution_text", "No detailed solution provided.")
            calculation_result = f"Type: {result_type}\nSolution: {result_text}"
        else:
            calculation_result = "Problem was not recognized. Please write clearly."
            result_type = "Error"
            result_text = calculation_result

        # UPDATE UI STATE
        app_state['analyzing'] = False
        app_state['problem'] = current_expression
        app_state['solution_type'] = result_type
        app_state['solution_text'] = result_text

        print(f"Model Read: {current_expression}. Solution Type: {api_response.get('solution_type')}")
        if calculation_result:
            print("\n--- DETAILED SOLUTION (Full Markdown/LaTeX) ---")
            print(result_text)
            print("------------------------------------------\n")

    display_canvas = cv2.resize(drawing_canvas, (w, h))
    gray_canvas = cv2.cvtColor(display_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    image = cv2.add(image, display_canvas)

    status_text = f"DRAW: {'ON' if drawing_active else 'OFF'} (Press 's' to toggle)"
    cv2.putText(image, status_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2, cv2.LINE_AA)
    
    exp_color = (0, 255, 0) if calculation_result and "Error" not in current_expression else (255, 255, 0)
    problem_display = "Problem: " + current_expression
    cv2.putText(image, problem_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, exp_color, 2, cv2.LINE_AA)
    
    if calculation_result is not None:
        solution_lines = calculation_result.split('\n')
        cv2.putText(image, solution_lines[0], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        if len(solution_lines) > 1:
            full_solution_text = " ".join(solution_lines[1:]).strip()
            solution_prefix_index = full_solution_text.find("Solution:")
            if solution_prefix_index != -1:
                display_solution = full_solution_text[solution_prefix_index + len("Solution:"):].strip()
            else:
                display_solution = full_solution_text.strip()
            
            if len(display_solution) > 50:
                display_solution = display_solution[:47] + "..."
            
            cv2.putText(image, f"Solution: {display_solution}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Gesture-Controlled Calculator', image)

    key = cv2.waitKey(5) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    
    if key == ord('c') or key == ord('C'):
        drawing_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        current_expression = "Draw a math problem (e.g., solve for x: 2x+3=7)"
        calculation_result = None
        # UPDATE UI STATE
        app_state['problem'] = current_expression
        app_state['solution_type'] = ''
        app_state['solution_text'] = ''
        print("Canvas cleared.")
    
    if key == ord('s') or key == ord('S'):
        drawing_active = not drawing_active
        # UPDATE UI STATE
        app_state['drawing_active'] = drawing_active
        print(f"Drawing mode: {'ACTIVATED' if drawing_active else 'DEACTIVATED'} (Key 's')")

cap.release()
cv2.destroyAllWindows()