# Air-Cursor

This is an air gesture based cursor using MediaPipe and OpenCV.

## Overview
Air-Cursor lets you control your computer mouse using your hand in front of the camera. It uses MediaPipe for hand tracking and OpenCV for the camera feed.

## Features
* Control mouse movement with your index finger
* Left click by pinching your index and thumb
* Scroll up with a thumbs up gesture
* Scroll down with a thumbs down gesture

## Requirements
To run this program you need to install these libraries and frameworks:

* OpenCV (for camera feed)
* MediaPipe (for hand tracking)
* PyAutoGUI (for mouse control)
* NumPy (for math operations)
* PyQt5 (for the user interface)
* Pynput (for secondary mouse control)

Use this command in your terminal to install everything:

pip install opencv-python mediapipe pyautogui numpy PyQt5 pynput

It is best to use Python 3.11 or newer.

## How to run
To run the program, use this command in your terminal:

python testhands.py

It takes a few seconds to initially launch so just wait for it to start.

## How to use
Point your index finger at the camera to move the cursor around the screen.

Pinch your index finger with your thumb to click on something.

Give a thumbs up to scroll up.

Give a thumbs down to scroll down.
