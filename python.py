import cv2
import mediapipe as mp
import numpy as np
import requests
import base64
import json
import time
import sys
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import threading
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ===== FLASK SETUP (MUST BE FIRST) =====
# Suppress Flask start message on console if running in certain environments
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)

# Updated app_state with new plotting field
app_state = {
    'drawing_active': True,
    'analyzing': False,
    'operation_mode': 'SOLVE', # New state
    'problem': 'Draw a math problem (e.g., solve for x: 2x+3=7)',
    'solution_type': '',
    'solution_text': '',
    'plottable_equation': '', # NEW: Holds the equation string for the frontend to plot
    'cooldown': False,
    'drawing_canvas_b64': '' # NEW: Base64 image of the drawing canvas
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/status')
def get_status():
    return jsonify(app_state)

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

print("\n" + "="*60)
print("🚀 FLASK SERVER STARTED! (http://localhost:5000)")
print("🔑 Use your webcam and the keyboard commands below:")
print("="*60 + "\n")
time.sleep(2) # Give Flask time to start

# ===== AI AND VISION CONFIG =====

CANVAS_SIZE = 500
LINE_COLOR = (255, 255, 255)
LINE_THICKNESS = 8
DOT_COLOR = (0, 0, 255)
SMOOTHING_FACTOR = 0.3
smoothed_point = None

# IMPORTANT: API Key is now empty string
# Load API key from environment variable
apiKey = os.getenv('API_KEY')
if not apiKey:
    raise ValueError("API_KEY not found in environment variables. Please add it to .env file.")
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
# Load API key from environment variable
apiKey = os.getenv('API_KEY')
FINGER_TIPS = [8, 12, 16, 20]
BASE_KNUCKLES = [5, 9, 13, 17]
PIP_JOINTS = [6, 10, 14, 18]

# --- Global State ---
drawing_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
last_point = None
drawing_active = True
cooldown_counter = 0
strokes = []         
redo_stack = []     
current_stroke = []
COOLDOWN_FRAMES = 30
current_expression = app_state['problem']
calculation_result = None
canvas_history = [] # For Undo functionality
MAX_HISTORY = 10 
is_new_stroke = False
operation_mode = 'SOLVE' # Default mode

# --- Gesture Detection Functions ---

def is_thumb_extended(landmarks):
    wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    thumb_tip_x = landmarks.landmark[THUMB_TIP_ID].x
    thumb_tip_y = landmarks.landmark[THUMB_TIP_ID].y
    thumb_base_x = landmarks.landmark[THUMB_BASE_ID].x
    thumb_base_y = landmarks.landmark[THUMB_BASE_ID].y
    
    # Simple check: is the thumb tip significantly farther from the wrist than the base?
    dist_tip_wrist = np.linalg.norm(np.array([thumb_tip_x, thumb_tip_y]) - np.array([wrist_x, wrist_y]))
    dist_base_wrist = np.linalg.norm(np.array([thumb_base_x, thumb_base_y]) - np.array([wrist_x, wrist_y]))
    
    return (dist_tip_wrist > dist_base_wrist + 0.05)

def is_calculate_gesture(hand_landmarks):
    # Gesture: Extended Thumb, all other fingers curled (a thumbs up / 'SOLVE' command)
    thumb_extended = is_thumb_extended(hand_landmarks)
    if not thumb_extended:
        return False
    
    all_fingers_curled = True
    for tip_id, base_id in zip(FINGER_TIPS, BASE_KNUCKLES):
        tip_y = hand_landmarks.landmark[tip_id].y
        base_y = hand_landmarks.landmark[base_id].y
        
        # Check if the tip is below the knuckle (curled)
        if tip_y < base_y:
            all_fingers_curled = False
            break
    
    return thumb_extended and all_fingers_curled

# --- API Functions ---

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def call_gemini_api(base64_image, operation_mode):
    time.sleep(0.5)
    
    # 1. Define Prompts based on the selected mode
    
    if operation_mode == 'DERIVATIVE':
        system_prompt = (
            "You are an expert calculus solver. Analyze the provided handwritten function. "
            "Your task is to identify the function, cleanly transcribe it, and provide the step-by-step first derivative with respect to x. "
            "Format the solution using Markdown, including LaTeX/math syntax for clarity."
        )
        prompt = "Read the function from this image and provide the result in the specified JSON format."
    elif operation_mode == 'INTEGRAL':
        system_prompt = (
            "You are an expert calculus solver. Analyze the provided handwritten function. "
            "Your task is to identify the function, cleanly transcribe it, and provide the step-by-step indefinite integral, including the constant of integration (+C). "
            "Format the solution using Markdown, including LaTeX/math syntax for clarity."
        )
        prompt = "Read the function from this image and provide the result in the specified JSON format."
    elif operation_mode == 'GRAPH':
        system_prompt = (
            "You are a geometric and algebraic interpreter. Analyze the provided handwritten equation/function. "
            "Your task is to identify the function, provide a description of its graph (e.g., shape, asymptotes, domain/range), and the key points (x and y intercepts). "
            "Crucially, you must also provide the function in a simple, single-variable JavaScript-executable string format (e.g., 'x * x', 'Math.sin(x)', '2 * x + 3') for plotting. Use 'Math.pow(x, N)' for powers, 'Math.sqrt()' for square roots, and always prefix trigonometric functions (e.g., 'Math.cos(x)'). "
            "Format the solution using Markdown, including LaTeX/math syntax for clarity."
        )
        prompt = "Read the equation/function from this image and provide the required graphing analysis and the plottable equation."
    else: # Default: SOLVE
        system_prompt = (
            "You are an expert mathematical solver and interpreter. Analyze the provided handwritten image. "
            "Your task is to identify the mathematical problem, cleanly transcribe it, and provide the step-by-step solution and final answer. "
            "Format the solution using Markdown, including LaTeX/math syntax for clarity. "
            "If you cannot confidently read a valid math problem, return 'INVALID' for all fields."
        )
        prompt = "Read the mathematical problem from this image and provide the result in the specified JSON format."

    # 2. Construct Payload
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
                    "solution_text": {"type": "STRING"},
                    # Plottable equation is included in all modes, but only useful for GRAPH mode
                    "plottable_equation": {"type": "STRING"} 
                }
            }
        }
    }

    # 3. Handle API Call with Exponential Backoff
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
        return {"problem": f"API Error: {err.response.status_code}", "solution_text": "Could not connect to the math solver.", "solution_type": "Error", "plottable_equation": ""}
    except Exception as e:
        return {"problem": f"API/Parsing Error: {e}", "solution_text": f"There was an issue processing the response: {e}", "solution_type": "Error", "plottable_equation": ""}


# --- Main Loop Setup ---

cap = cv2.VideoCapture(0)
cv2.namedWindow('Gesture-Controlled Calculator')

print("--- AI Handwriting Symbolic Math Solver ---")
print("Keyboard Commands:")
print(" 's' : Toggle Drawing ON/OFF")
print(" 'c' : Clear Canvas")
print(" 'b' : Undo Last Stroke")
print(" 'f' : Toggle Fullscreen")
print("\nOperation Modes:")
print(" 'z' : SOLVE (Default - Algebra/Basic Math)")
print(" 'd' : DERIVATIVE")
print(" 'i' : INTEGRAL")
print(" 'g' : GRAPH Analysis (New Plotting Feature!)")
print("\nGesture Command:")
print(" SOLVE (API Trigger): Hold up ONLY your THUMB.")
print("---------------------------------")

# Initial UI State Sync
app_state['operation_mode'] = operation_mode
app_state['plottable_equation'] = ''

# --- Main Video Processing Loop ---
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
        
        # Calculate raw finger point
        raw_x = int(hand_landmarks.landmark[INDEX_FINGER_TIP_ID].x * w)
        raw_y = int(hand_landmarks.landmark[INDEX_FINGER_TIP_ID].y * h)
        raw_finger_point = (raw_x, raw_y)

        # Apply smoothing
        if smoothed_point is None:
            smoothed_point = raw_finger_point
        else:
            smooth_x, smooth_y = smoothed_point
            new_smooth_x = int(smooth_x * (1 - SMOOTHING_FACTOR) + raw_x * SMOOTHING_FACTOR)
            new_smooth_y = int(smooth_y * (1 - SMOOTHING_FACTOR) + raw_y * SMOOTHING_FACTOR)
            smoothed_point = (new_smooth_x, new_smooth_y)
        
        current_point = smoothed_point
        
        # Check for Gesture
        if is_calculate_gesture(hand_landmarks):
            gesture_command = 'CALCULATE'
        
        # Draw cursor dot
        cv2.circle(image, current_point, 8, DOT_COLOR, -1)
    else:
    # finger lifted → complete stroke
        if len(current_stroke) > 0:
            strokes.append(current_stroke.copy())
            current_stroke.clear()
            redo_stack.clear()   # clear redo because new stroke overrides it
        last_point = None
        smoothed_point = None # Ensure new stroke flag is reset when hand leaves frame

    # --- Drawing Logic ---
    if drawing_active and current_point:
        if last_point is None:
            # START OF NEW STROKE: Save current canvas state to history
            is_new_stroke = True
            if len(canvas_history) >= MAX_HISTORY:
                canvas_history.pop(0) # Remove oldest
            canvas_history.append(drawing_canvas.copy())
            
        if last_point:
            # Map point from camera resolution to canvas resolution
            cv_x1 = int(last_point[0] * (CANVAS_SIZE / w))
            cv_y1 = int(last_point[1] * (CANVAS_SIZE / h))
            cv_x2 = int(current_point[0] * (CANVAS_SIZE / w))
            cv_y2 = int(current_point[1] * (CANVAS_SIZE / h))
            
            # Draw the line segment
            cv2.line(drawing_canvas, (cv_x1, cv_y1), (cv_x2, cv_y2), LINE_COLOR, LINE_THICKNESS)
            current_stroke.append((cv_x2, cv_y2))

            
        last_point = current_point
    else:
        last_point = None
        
    # --- Cooldown and Gesture Trigger ---
    if cooldown_counter > 0:
        cooldown_counter -= 1
        app_state['cooldown'] = True
        gesture_command = None
    else:
        app_state['cooldown'] = False


    if gesture_command == 'CALCULATE' and cooldown_counter == 0:
        print(f"\n--- API Command Received: {operation_mode} ---")
        current_expression = f"CAPTURING & ANALYZING SYMBOLIC MATH ({operation_mode})..."
        calculation_result = None
        cooldown_counter = COOLDOWN_FRAMES
        
        # UPDATE UI STATE (Start Analysis)
        app_state['analyzing'] = True
        app_state['problem'] = current_expression
        app_state['plottable_equation'] = '' # Clear old plot
        
        # Run API Call
        temp_img = drawing_canvas.copy()
        base64_img = image_to_base64(temp_img)
        api_response = call_gemini_api(base64_img, operation_mode)
        
        # Process API Response
        current_expression = api_response.get("problem", "Could not read problem.")
        result_type = api_response.get("solution_type", operation_mode)
        result_text = api_response.get("solution_text", "No detailed solution provided.")
        plottable_eq = api_response.get("plottable_equation", "") # Get the new plottable string

        if current_expression == "INVALID":
            result_type = "Error"
            result_text = "Problem was not recognized. Please write clearly."

        # Update global result variables
        calculation_result = result_text 

        # UPDATE UI STATE (End Analysis)
        app_state['analyzing'] = False
        app_state['problem'] = current_expression
        app_state['solution_type'] = result_type
        app_state['solution_text'] = result_text
        app_state['plottable_equation'] = plottable_eq # Set the new plottable string

        print(f"Model Read: {current_expression}. Solution Type: {result_type}. Plottable Eq: {plottable_eq}")
        print("------------------------------------------")

    # --- Display Canvas Overlay ---
    display_canvas = cv2.resize(drawing_canvas, (w, h))
    image = cv2.add(image, display_canvas)

    # Convert the current drawing canvas to Base64 for the frontend
    app_state['drawing_canvas_b64'] = image_to_base64(drawing_canvas)

    # --- Key Press Handlers ---
    key = cv2.waitKey(5) & 0xFF

    if key == ord('u') or key == ord('U'):
        if strokes:
            redo_stack.append(strokes.pop())
            drawing_canvas[:] = 0  # clear canvas
            for stroke in strokes:
                for i in range(1, len(stroke)):
                    cv2.line(drawing_canvas, stroke[i-1], stroke[i], LINE_COLOR, LINE_THICKNESS)
            print("Undo performed.")
    
    if key == ord('y') or key == ord('Y'):
        if redo_stack:
            restored = redo_stack.pop()
            strokes.append(restored)
            for i in range(1, len(restored)):
                cv2.line(drawing_canvas, restored[i-1], restored[i], LINE_COLOR, LINE_THICKNESS)
            print("Redo performed.")
    if key == ord('q') or key == ord('Q'):
        break
    
    elif key == ord('c') or key == ord('C'):
        drawing_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        canvas_history = []
        current_expression = f"Canvas cleared. Mode: {operation_mode}"
        calculation_result = None
        # UPDATE UI STATE
        app_state['problem'] = current_expression
        app_state['solution_type'] = ''
        app_state['solution_text'] = ''
        app_state['plottable_equation'] = ''
        print("Canvas cleared.")
    
    elif key == ord('s') or key == ord('S'):
        drawing_active = not drawing_active
        # UPDATE UI STATE
        app_state['drawing_active'] = drawing_active
        print(f"Drawing mode: {'ACTIVATED' if drawing_active else 'DEACTIVATED'} (Key 's')")
        
    elif key == ord('z') or key == ord('Z'):
        operation_mode = 'SOLVE'
        current_expression = "MODE: SOLVE. Draw an algebraic equation."
        app_state['operation_mode'] = operation_mode
        app_state['plottable_equation'] = ''
        print(f"Mode set to: {operation_mode}")
        
    elif key == ord('d') or key == ord('D'):
        operation_mode = 'DERIVATIVE'
        current_expression = "MODE: DERIVATIVE. Draw a function to differentiate."
        app_state['operation_mode'] = operation_mode
        app_state['plottable_equation'] = ''
        print(f"Mode set to: {operation_mode}")
        
    elif key == ord('i') or key == ord('I'):
        operation_mode = 'INTEGRAL'
        current_expression = "MODE: INTEGRAL. Draw a function to integrate."
        app_state['operation_mode'] = operation_mode
        app_state['plottable_equation'] = ''
        print(f"Mode set to: {operation_mode}")
        
    elif key == ord('g') or key == ord('G'):
        operation_mode = 'GRAPH'
        current_expression = "MODE: GRAPH. Draw an equation to analyze/plot."
        app_state['operation_mode'] = operation_mode
        app_state['plottable_equation'] = ''
        print(f"Mode set to: {operation_mode}")
        
    elif key == ord('b') or key == ord('B'): # Backspace/Undo
        if len(canvas_history) > 0:
            drawing_canvas = canvas_history.pop()
            current_expression = "Undo successful."
        else:
            current_expression = "Cannot undo, history is empty."
        app_state['problem'] = current_expression
        app_state['solution_type'] = '' # Clear old result on undo
        app_state['solution_text'] = ''
        app_state['plottable_equation'] = ''
        
    elif key == ord('f') or key == ord('F'): # Fullscreen toggle
        prop = cv2.getWindowProperty('Gesture-Controlled Calculator', cv2.WND_PROP_FULLSCREEN)
        if prop < 1:
            cv2.setWindowProperty('Gesture-Controlled Calculator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Gesture-Controlled Calculator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # --- OpenCV Text Overlay for User Feedback ---
    
    # Status text
    status_text = f"DRAW: {'ON' if drawing_active else 'OFF'} (s) | Mode: {operation_mode} (z/d/i/g)"
    cv2.putText(image, status_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2, cv2.LINE_AA)
    
    # Problem/Action text
    exp_color = (0, 255, 0) if "CAPTURING" not in current_expression else (255, 255, 0)
    problem_display = "Action: " + current_expression
    
    # Only display first part of the problem on the camera feed
    if len(problem_display) > 80:
        problem_display = problem_display[:77] + "..."

    cv2.putText(image, problem_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, exp_color, 2, cv2.LINE_AA)
    
    # Solution preview (just show type for quick feedback)
    if app_state['solution_type'] and app_state['solution_text']:
        solution_display = f"Result Type: {app_state['solution_type']}"
        cv2.putText(image, solution_display, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Gesture-Controlled Calculator', image)

cap.release()
cv2.destroyAllWindows()