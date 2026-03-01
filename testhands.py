import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import sys
import os
import math
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QFont, QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from pynput.mouse import Controller, Button

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo", "download.jpg")
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class CameraWorker(QThread):
    # Signals to update UI from background thread
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.mouse = Controller()
        self.smooth_x, self.smooth_y = 0, 0
        self.click_down = False

    def run(self):
        self.status_signal.emit("Initializing Camera...")
        
        # 1. ROBUST CAMERA DISCOVERY
        # Try index 0 and 1 (Macs often hide cam on 1)
        cap = None
        for index in [0, 1]:
            self.status_signal.emit(f"Testing Camera Index {index}...")
            temp_cap = cv2.VideoCapture(index)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret and frame is not None:
                    cap = temp_cap
                    self.status_signal.emit(f"Camera Found at Index {index}")
                    break
                temp_cap.release()
        
        if cap is None:
            self.status_signal.emit("ERROR: No Camera Found.\nCheck Privacy Permissions.")
            return

        # 2. TRACKING LOOP
        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        ) as hands:
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.status_signal.emit("Frame Drop / Camera Disconnected")
                    time.sleep(0.1)
                    continue

                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # Process Hands
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Logic
                    index_tip = hand.landmark[8]
                    thumb_tip = hand.landmark[4]
                    
                    # --- MOVEMENT LOGIC (SMOOTHING) ---
                    margin = 0.15  # 15% margin for easier edge reach
                    x = (index_tip.x - margin) / (1 - 2 * margin)
                    y = (index_tip.y - margin) / (1 - 2 * margin)
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    
                    target_px = int(x * SCREEN_WIDTH)
                    target_py = int(y * SCREEN_HEIGHT)
                    
                    # --- CLICK LOGIC (INDEX-THUMB PINCH) ---
                    pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    
                    # Click thresholds
                    click_threshold = 0.04
                    release_threshold = 0.05
                    
                    if pinch_dist < click_threshold and not self.click_down:
                        self.mouse.press(Button.left)
                        self.click_down = True
                    elif pinch_dist > release_threshold and self.click_down:
                        self.mouse.release(Button.left)
                        self.click_down = False

                    # --- SCROLL LOGIC (THUMBS UP/DOWN) ---
                    # Check if other fingers are folded
                    # Tips: 8, 12, 16, 20. PIPs: 6, 10, 14, 18
                    fingers_folded = all(hand.landmark[i].y > hand.landmark[i-2].y for i in [8, 12, 16, 20])
                    
                    scrolling = False
                    if fingers_folded:
                        # Thumb Tip (4) vs Thumb IP (3)
                        thumb_up = hand.landmark[4].y < hand.landmark[3].y
                        thumb_down = hand.landmark[4].y > hand.landmark[3].y
                        
                        # Use a small distance threshold to ensure it's a clear vertical thumb
                        if abs(hand.landmark[4].y - hand.landmark[3].y) > 0.05:
                            if thumb_up:
                                pyautogui.scroll(10)
                                cv2.putText(frame, "SCROLL UP", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                scrolling = True
                            elif thumb_down:
                                pyautogui.scroll(-10)
                                cv2.putText(frame, "SCROLL DOWN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                scrolling = True

                    # --- EXECUTE MOVE (STABILIZED) ---
                    # Freeze cursor position while clicking or scrolling to prevent jitter
                    if not self.click_down and not scrolling:
                        # Dynamic Smoothing: Fast move = responsive, Slow move = smooth
                        curr_dist = math.hypot(target_px - self.smooth_x, target_py - self.smooth_y)
                        alpha = np.interp(curr_dist, [5, 200], [0.03, 0.4])
                        
                        self.smooth_x += alpha * (target_px - self.smooth_x)
                        self.smooth_y += alpha * (target_py - self.smooth_y)
                        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))
                    
                    # --- VISUAL FEEDBACK (ON INDEX TIP) ---
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    color = (0, 0, 255) if self.click_down else (0, 255, 0)
                    # Larger indicator on the pointer
                    cv2.circle(frame, (ix, iy), 15, color, -1 if self.click_down else 2)
                    if not self.click_down and not scrolling:
                        cv2.circle(frame, (ix, iy), 5, (255, 255, 255), -1) # Small center dot

                # Send frame to UI
                self.frame_signal.emit(frame)
        
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.thread = None

    def initUI(self):
        self.setWindowTitle("Air-Cursor Pro")
        self.setFixedSize(660, 580)
        
        # Set Icon for Window
        if os.path.exists(LOGO_PATH):
            self.setWindowIcon(QIcon(LOGO_PATH))
        
        layout = QVBoxLayout()
        
        # Camera Feed / Status Area
        self.feed_label = QLabel("Click 'Start Camera' to begin", self)
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setStyleSheet("background-color: #222; color: #EEE; font-size: 14px; border-radius: 8px;")
        self.feed_label.setFixedSize(640, 480)
        layout.addWidget(self.feed_label, alignment=Qt.AlignCenter)
        
        # Controls
        self.btn_start = QPushButton("Start Camera", self)
        self.btn_start.setFixedSize(640, 50)
        self.btn_start.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_cam)
        layout.addWidget(self.btn_start, alignment=Qt.AlignCenter)
        
        self.btn_quit = QPushButton("Quit", self)
        self.btn_quit.setFixedSize(640, 40)
        self.btn_quit.setStyleSheet("background-color: #c62828; color: white; border-radius: 5px;")
        self.btn_quit.clicked.connect(self.close)
        layout.addWidget(self.btn_quit, alignment=Qt.AlignCenter)
        
        self.setLayout(layout)

    def start_cam(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Index to move cursor, Index+Thumb to click")
        
        self.thread = CameraWorker()
        self.thread.frame_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    @pyqtSlot(str)
    def update_status(self, text):
        self.feed_label.setText(text)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        

        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.feed_label.width(), 
            self.feed_label.height(), 
            Qt.KeepAspectRatio
        )
        self.feed_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()

if __name__ == "__main__":
    print("Launching Program Can take 10 seconds - Ifrad ")
    app = QApplication(sys.argv)
    
    # FORCE DOCK ICON (macOS specific)
    if os.path.exists(LOGO_PATH):
        app.setWindowIcon(QIcon(LOGO_PATH))
    
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())