Math on Fingertips — Air-Writing Finger Tracking Math Solver

This project lets you write math expressions in the air using your index finger, captures the strokes through real-time hand tracking, and sends the drawn equation to an AI backend for solving, graphing, and step-by-step explanations.

The project combines computer vision, gesture tracking, and AI reasoning to build an interface that feels both intuitive and futuristic.

Inspiration

This idea started when I wanted a more natural and playful way to interact with math problems. Typing formulas feels strict and slow.
Drawing them in space with your fingertip felt more expressive.

However, building stable fingertip tracking turned out to be far more complex than I expected.
This project became a deep exploration into:

how hand-landmark models behave in real-time,

how to stabilize noisy fingertip predictions,

how to merge strokes correctly, and

how to handle the imperfections of camera input.

The technical journey—especially fixing jitter, misdetections, and tracking gaps—shaped the project into what it is now.

How Finger Tracking Works
1. Webcam Input

OpenCV streams the camera frames at real-time speed.
Each frame is processed immediately and fed into a hand-tracking model.

2. MediaPipe Hands Landmark Model

MediaPipe predicts 21 landmarks per hand, including:

index finger tip

knuckles

palm center

wrist

The model is robust, but not perfect. It can:

flicker under poor lighting

lose tracking when the finger rotates

mis-detect depth

momentarily jump between frames

3. Extracting the Index Fingertip

The fingertip is landmark 8.
Every frame:

If a hand is detected, the coordinates (x, y) are read.

These coordinates are scaled to the screen resolution.

4. Stabilizing the Finger Path

Raw fingertip coordinates are noisy.
To fix that, the system uses:

Smoothing Filter

A moving-average or exponential smoothing filter reduces jitter, so strokes look continuous instead of shaky.

Without smoothing:

lines jump

curves look broken

strokes become messy

With smoothing:

the fingertip appears steady

writing becomes readable

5. Stroke Building

The program builds strokes by tracking:

when a finger is “down”

when it lifts

how fast it is moving

how far two consecutive points are from each other

Strokes are stored as arrays of points.
This enables:

undo

redo

clearing

sending the entire stroke history to the backend

6. Challenges Solved

The hand tracking layer required solving multiple issues:

Jitter: fixed with smoothing filters

False detections: removed with distance checks and frame-consistency tests

Missing frames: interpolated to keep strokes continuous

Lag from model inference: optimized by reducing resolution

Finger rotating away from the camera: handled with fallback detection

Stroke cutting: fixed by tracking “writing state” vs “idle state”

Much of the project’s complexity came from stabilizing these behaviors.

Backend Overview

The backend is a Python Flask server that:

receives stroke data

converts strokes into an equation image

sends a structured prompt to the AI model

handles step-by-step solutions

computes derivatives or integrals

generates graphs

returns everything to the frontend as JSON

This backend runs exactly as it does locally and can be deployed on Render.

Frontend Overview

The frontend is an HTML + JS interface that:

displays the canvas

shows the hand-drawn strokes

formats the returned math

shows graphs

provides keyboard shortcuts for:

G (graph)

D (derivative)

I (integral)

Z (clear)

F (fullscreen)

The front and back communicate through a simple fetch API.

Tech Stack
Computer Vision

OpenCV

MediaPipe Hands

Custom smoothing filters

Stroke segmentation logic

Backend

Python

Flask

Matplotlib

AI model API

Frontend

HTML

CSS

JavaScript

MathJax

Features

Real-time fingertip tracking

Smooth stroke rendering

Undo and redo

Drawing in the air

Automatic equation solving

Graph generation

Step-by-step explanations

Keyboard shortcuts

Fullscreen canvas mode

Installation

Install dependencies:

pip install flask flask-cors opencv-python mediapipe matplotlib google-generativeai


Run backend:

python python.py


Open index.html in a browser.

Why This Project Is Not Fully Deployable

Web servers cannot access user webcams from Python.
This means real-time fingertip tracking must run on the user's local device.
The backend can be deployed, but the computer vision layer must stay client-side.


Conclusion

This project became a deep dive into stabilizing real-time hand tracking.
Making fingertip movements smooth, readable, and consistent was the biggest challenge—and the most rewarding part.
It demonstrates how computer vision systems require careful handling of noise, jitter, and imperfect predictions, and how combining them with modern AI can produce a uniquely interactive experience.
