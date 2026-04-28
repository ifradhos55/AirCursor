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

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo", "download.jpg")
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

class CameraWorker(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.mouse = Controller()
        self.smooth_x, self.smooth_y = 0, 0
        self.click_down = False
        self.mission_control_locked = False

    def run(self):
        self.status_signal.emit("Initializing Camera...")
        cap = None
        for index in [0, 1]:
            temp_cap = cv2.VideoCapture(index)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret and frame is not None:
                    cap = temp_cap
                    break
                temp_cap.release()
        
        if cap is None:
            self.status_signal.emit("ERROR: No Camera Found.")
            return

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        ) as hands:
            
            while self.running:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Logic
                    index_tip = hand.landmark[8]
                    thumb_tip = hand.landmark[4]
                    
                    # --- MOVEMENT LOGIC (SMOOTHING) ---
                    margin = 0.15  
                    x = np.clip((index_tip.x - margin) / (1 - 2 * margin), 0, 1)
                    y = np.clip((index_tip.y - margin) / (1 - 2 * margin), 0, 1)
                    target_px = int(x * SCREEN_WIDTH)
                    target_py = int(y * SCREEN_HEIGHT)
                    
                    # --- CLICK LOGIC (PINCH) ---
                    pinch_dist = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                    if pinch_dist < 0.04 and not self.click_down:
                        self.mouse.press(Button.left)
                        self.click_down = True
                    elif pinch_dist > 0.05 and self.click_down:
                        self.mouse.release(Button.left)
                        self.click_down = False

                    # --- SCROLL LOGIC (THUMBS UP/DOWN) ---
                    fingers_folded = all(hand.landmark[i].y > hand.landmark[i-2].y for i in [8, 12, 16, 20])
                    scrolling = False
                    if fingers_folded:
                        thumb_y_diff = hand.landmark[4].y - hand.landmark[3].y
                        if abs(thumb_y_diff) > 0.05:
                            if thumb_y_diff < 0: # Thumb Up
                                pyautogui.scroll(7)
                                cv2.putText(frame, "SCROLL UP", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                scrolling = True
                            else: # Thumb Down
                                pyautogui.scroll(-7)
                                cv2.putText(frame, "SCROLL DOWN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                scrolling = True

                    # --- MISSION CONTROL (4 FINGERS) ---
                    four_fingers_up = all(hand.landmark[i].y < hand.landmark[i-2].y for i in [8, 12, 16, 20])
                    if four_fingers_up:
                        if not self.mission_control_locked:
                            if sys.platform == "darwin":
                                pyautogui.hotkey('ctrl', 'up')
                            else:
                                pyautogui.hotkey('win', 'tab')
                            self.mission_control_locked = True
                    else:
                        self.mission_control_locked = False

                    if self.mission_control_locked:
                         cv2.putText(frame, "MISSION CONTROL", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                    # --- EXECUTE MOVE ---
                    if not self.click_down and not scrolling:
                        curr_dist = math.hypot(target_px - self.smooth_x, target_py - self.smooth_y)
                        alpha = np.interp(curr_dist, [5, 200], [0.03, 0.4])
                        self.smooth_x += alpha * (target_px - self.smooth_x)
                        self.smooth_y += alpha * (target_py - self.smooth_y)
                        pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))
                    
                    # --- VISUAL FEEDBACK ---
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(frame, (ix, iy), 15, (0, 0, 255) if self.click_down else (0, 255, 0), -1 if self.click_down else 2)

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
        self.setFixedSize(660, 600)
        if os.path.exists(LOGO_PATH): self.setWindowIcon(QIcon(LOGO_PATH))
        
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #e0e0e0; font-family: 'Arial'; }
            #feed_label { background-color: #000; border: 2px solid #333; border-radius: 15px; }
            QPushButton { background-color: #1b5e20; color: white; border-radius: 8px; font-weight: bold; height: 40px; }
            QPushButton:hover { background-color: #2e7d32; }
            #quit_btn { background-color: #b71c1c; }
            #quit_btn:hover { background-color: #c62828; }
        """)

        layout = QVBoxLayout()
        self.feed_label = QLabel("Click Start to begin", self)
        self.feed_label.setObjectName("feed_label")
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setFixedSize(640, 480)
        layout.addWidget(self.feed_label, alignment=Qt.AlignCenter)

        self.btn_start = QPushButton("START AIR-CURSOR", self)
        self.btn_start.clicked.connect(self.start_cam)
        layout.addWidget(self.btn_start)

        self.btn_quit = QPushButton("EXIT", self)
        self.btn_quit.setObjectName("quit_btn")
        self.btn_quit.clicked.connect(self.close)
        layout.addWidget(self.btn_quit)

        self.setLayout(layout)

    def start_cam(self):
        self.btn_start.setEnabled(False)
        self.thread = CameraWorker()
        self.thread.frame_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

    @pyqtSlot(str)
    def update_status(self, text): self.feed_label.setText(text)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(640, 480, Qt.KeepAspectRatio)
        self.feed_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if self.thread: self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())