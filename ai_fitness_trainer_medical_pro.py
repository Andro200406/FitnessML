# ai_fitness_trainer_medical_pro.py
# Single-file: AI Fitness Trainer (Desktop) with medical estimations, form correction, CSV logging, and video recording.
# Run as: python ai_fitness_trainer_medical_pro.py

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import csv
from collections import deque
from threading import Thread
from queue import Queue
from datetime import datetime
import math
import statistics
import os
import sys

# ----------------------- Configuration -----------------------
EXERCISE_CONFIG = {
    "Straight Posture": {"angle_range": (165, 180)},
    "Head Rotation": {"turn_threshold": 0.14},
    "Jumping Jacks": {"min_open": 150, "min_close": 90},
    "Push-ups": {"up_min": 150, "down_max": 90},
    "Pull-ups": {"up_max": 80, "down_min": 150},
    "Squats": {"up_min": 160, "down_max": 85},
    "Lunges": {"up_min": 160, "down_max": 85},
    "Bicep Curls": {"down_min": 150, "up_max": 50},
    "Leg Raise": {"down_min": 150, "up_max": 90},
    "Burpees": {"stand_min": 160, "down_max": 85},
    "Plank": {"angle_range": (160, 180)},
    "Walking": {"step_mag": 0.015},
    "Calf Raises": {"up_min": 8, "down_max": 4}
}

EXERCISE_LIST = list(EXERCISE_CONFIG.keys())
ANGLE_SMOOTHING = 6
FPS_SMOOTHING = 10

# rPPG & physiology params
RPPG_BUFFER_SECONDS = 12
RPPG_SAMPLE_FPS = 30
BREATH_BUFFER_SECONDS = 20
HR_COMPUTE_INTERVAL = 2.0
BR_COMPUTE_INTERVAL = 3.0
SP02_HEURISTIC_FACTOR = 0.92
TEMP_BASELINE = 36.6

HYDRATION_INTERVAL_SECONDS = 10 * 60  # 10 minutes

# ----------------------- Helpers -----------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    # angle at b between vectors ba and bc
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_val))
    return float(angle)

def to_pixel_coords(landmark, image_shape):
    h, w = image_shape[:2]
    if hasattr(landmark, "x"):
        return int(landmark.x * w), int(landmark.y * h)
    else:
        return int(landmark[0] * w), int(landmark[1] * h)

def safe_mean(img):
    if img is None or img.size == 0:
        return 0.0
    return float(np.mean(img))

def next_video_filename(prefix="video", ext="avi"):
    # find next available video{n}.avi
    i = 1
    while True:
        name = f"{prefix}{i}.{ext}"
        if not os.path.exists(name):
            return name
        i += 1

# --------------------- TTS Worker ---------------------
class TTSWorker:
    def __init__(self, enabled=False):
        self.queue = Queue()
        self.engine = None
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
        except Exception:
            self.engine = None
        self.enabled = enabled and (self.engine is not None)
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            text = self.queue.get()
            if text is None:
                break
            if self.enabled and self.engine:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass

    def speak(self, text):
        if self.enabled:
            self.queue.put(text)

    def toggle(self):
        if self.engine is None:
            return False
        self.enabled = not self.enabled
        return self.enabled

    def stop(self):
        self.queue.put(None)

# --------------------- Exercise Detector (enhanced) ---------------------
class ExerciseDetector:
    
    
    def __init__(self, tts_worker=None, weight_kg=70, height_m=1.75, age=30):
        # mediapipe models
        self.mp_pose = mp.solutions.pose
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                                               min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.tts = tts_worker or TTSWorker(enabled=False)

        # Exercise counters & states
        self.counters = {ex: 0 for ex in EXERCISE_LIST}
        self.stages = {ex: None for ex in EXERCISE_LIST}
        self.angle_buffers = {ex: deque(maxlen=ANGLE_SMOOTHING) for ex in EXERCISE_LIST}
        self.last_speech_time = 0
        self.speech_cooldown = 0.6
        self.session_log = []

        # Fitness & medical
        self.start_time = None
        self.total_time = 0.0
        self.calories_burned = 0.0
        self.heart_rate = 72.0
        self.hr_buffer = deque()
        self.rppg_buffer = deque()
        self.rppg_t0 = None
        self.last_hr_compute = 0.0

        self.breath_buffer = deque()
        self.last_br_compute = 0.0
        self.breath_rate = 12.0

        self.spO2 = 98.0
        self.skin_temp = TEMP_BASELINE
        self.bp_systolic = 120
        self.bp_diastolic = 80

        self.intensity = "Low"
        self.weight_kg = weight_kg
        self.height_m = height_m
        self.age = age
        self.bmi = self._compute_bmi()
        self.last_hydration = time.time()
        self.hydration_reminder_flag = False

        # HRV / stress/fatigue
        self.hr_time_series = deque(maxlen=300)
        self.fatigue_index = 0.0
        self.stress_index = 0.0

        # walking/calf internal state
        self.prev_hip_y = None
        self.prev_ankle_y = None
        

    def _compute_bmi(self):
        try:
            return self.weight_kg / (self.height_m ** 2)
        except Exception:
            return 0.0

    def estimate_calories(self, exercise, reps, duration_sec):
        MET_VALUES = {
            "Straight Posture": 1.8,
            "Head Rotation": 1.5,
            "Jumping Jacks": 8.0,
            "Push-ups": 7.0,
            "Pull-ups": 9.0,
            "Squats": 6.0,
            "Lunges": 5.5,
            "Bicep Curls": 3.8,
            "Leg Raise": 4.0,
            "Burpees": 8.5,
            "Plank": 3.3,
            "Walking": 3.5,
            "Calf Raises": 4.0
        }
        met = MET_VALUES.get(exercise, 3.0)
        hours = duration_sec / 3600.0 if duration_sec > 0 else 0.0
        return met * self.weight_kg * hours
    
    def get_live_metrics(self, exercise):
        return {
            "exercise": exercise,
            "reps": self.counters.get(exercise, 0),
            "calories": round(self.calories_burned, 2),
            "heart_rate": int(self.heart_rate),
            "breath_rate": round(self.breath_rate, 1),
            "spo2": round(self.spO2, 1),
            "skin_temp": round(self.skin_temp, 2),
            "bp": f"{self.bp_systolic}/{self.bp_diastolic}",
            "intensity": self.intensity,
            "fatigue": round(self.fatigue_index, 1),
            "stress": round(self.stress_index, 1)
        }

    def update_fitness_factors(self, exercise):
        self.total_time = time.time() - (self.start_time or time.time())
        reps = self.counters.get(exercise, 0)
        self.calories_burned = self.estimate_calories(exercise, reps, self.total_time)
        hr = self.heart_rate
        if hr < 100:
            self.intensity = "Low"
        elif hr < 130:
            self.intensity = "Moderate"
        else:
            self.intensity = "High"
        self.bp_systolic = int(90 + 0.6 * hr + 0.4 * max(0, (100 - self.spO2)))
        self.bp_diastolic = int(60 + 0.25 * hr + 0.2 * max(0, (100 - self.spO2)))

        if len(self.hr_time_series) >= 5:
            hr_std = statistics.pstdev(self.hr_time_series)
            self.fatigue_index = max(0.0, min(100.0, 50.0 - hr_std * 5.0))
            br = self.breath_rate or 12.0
            self.stress_index = max(0.0, min(100.0, (40.0 / (hr_std + 0.1)) + abs(br - 12.0)))
        else:
            self.fatigue_index = 0.0
            self.stress_index = 0.0

        if (time.time() - self.last_hydration) > HYDRATION_INTERVAL_SECONDS and (self.intensity == "High" or self.total_time > 600):
            self.hydration_reminder_flag = True
        else:
            self.hydration_reminder_flag = False

    # rPPG sampling (forehead ROI estimated using face landmarks)
    def compute_rppg_from_face(self, frame):
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return
            face_landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            # forehead-ish indices (MediaPipe face mesh approximate indices)
            idxs = [10, 338, 127, 356, 234, 10]
            xs, ys = [], []
            for i in idxs:
                if i < len(face_landmarks):
                    lm = face_landmarks[i]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))
            if not xs:
                return
            xmin, xmax = max(0, min(xs)-8), min(w-1, max(xs)+8)
            ymin, ymax = max(0, min(ys)-8), min(h-1, max(ys)+8)
            if xmax - xmin < 6 or ymax - ymin < 6:
                return
            roi = frame[ymin:ymax, xmin:xmax]
            mean_g = safe_mean(roi[:, :, 1])
            mean_r = safe_mean(roi[:, :, 2])
            ts = time.time()
            if self.rppg_t0 is None:
                self.rppg_t0 = ts
            self.rppg_buffer.append((ts, mean_g, mean_r))
            while len(self.rppg_buffer) > 0 and (self.rppg_buffer[-1][0] - self.rppg_buffer[0][0]) > RPPG_BUFFER_SECONDS:
                self.rppg_buffer.popleft()
        except Exception:
            return

    def compute_hr_from_rppg(self):
        if len(self.rppg_buffer) < 6:
            return
        now = time.time()
        if now - self.last_hr_compute < HR_COMPUTE_INTERVAL:
            return
        self.last_hr_compute = now
        times = np.array([t for (t, g, r) in self.rppg_buffer])
        greens = np.array([g for (t, g, r) in self.rppg_buffer])
        duration = times[-1] - times[0] if len(times) > 1 else 0.0
        if duration < 2.0:
            return
        fs = RPPG_SAMPLE_FPS
        num = min(int(duration * fs), max(6, len(greens)))
        try:
            interp_times = np.linspace(times[0], times[-1], num=num)
            greens_interp = np.interp(interp_times, times, greens)
            greens_d = greens_interp - np.mean(greens_interp)
            fft = np.fft.rfft(greens_d * np.hamming(len(greens_d)))
            freqs = np.fft.rfftfreq(len(greens_d), d=1.0/fs)
            mask = (freqs >= 0.7) & (freqs <= 3.5)
            if not np.any(mask):
                return
            p = np.abs(fft[mask])
            if p.size == 0:
                return
            idx = np.argmax(p)
            dominant_freq = freqs[mask][idx]
            hr_bpm = dominant_freq * 60.0
            # smoothing to reduce jumps
            self.heart_rate = 0.75 * self.heart_rate + 0.25 * hr_bpm
            self.hr_buffer.append((time.time(), self.heart_rate))
            self.hr_time_series.append(self.heart_rate)
            if len(self.hr_buffer) > 300:
                self.hr_buffer.popleft()
        except Exception:
            return

    # Breathing rate from chest vertical oscillation
    def append_breath_sample(self, landmarks_pose, image_shape):
        try:
            left_sh = landmarks_pose[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_sh = landmarks_pose[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = landmarks_pose[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks_pose[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            shoulder_y = (left_sh.y + right_sh.y) / 2.0
            hip_y = (left_hip.y + right_hip.y) / 2.0
            chest_y_norm = (shoulder_y + 0.25 * hip_y) / 1.25
            ts = time.time()
            self.breath_buffer.append((ts, chest_y_norm))
            while len(self.breath_buffer) > 0 and (self.breath_buffer[-1][0] - self.breath_buffer[0][0]) > BREATH_BUFFER_SECONDS:
                self.breath_buffer.popleft()
        except Exception:
            pass

    def compute_breath_rate(self):
        if len(self.breath_buffer) < 6:
            return
        now = time.time()
        if now - self.last_br_compute < BR_COMPUTE_INTERVAL:
            return
        self.last_br_compute = now
        times = np.array([t for (t, y) in self.breath_buffer])
        ys = np.array([y for (t, y) in self.breath_buffer])
        duration = times[-1] - times[0]
        if duration < 6.0:
            return
        try:
            fs = 10.0
            num = int(duration * fs)
            interp_times = np.linspace(times[0], times[-1], num=num)
            ys_interp = np.interp(interp_times, times, ys)
            ys_d = ys_interp - np.mean(ys_interp)
            fft = np.fft.rfft(ys_d * np.hamming(len(ys_d)))
            freqs = np.fft.rfftfreq(len(ys_d), d=1.0/fs)
            mask = (freqs >= 0.08) & (freqs <= 0.7)
            if not np.any(mask):
                return
            p = np.abs(fft[mask])
            idx = np.argmax(p)
            dominant = freqs[mask][idx]
            br_bpm = dominant * 60.0
            self.breath_rate = 0.6 * self.breath_rate + 0.4 * br_bpm
        except Exception:
            return

    def compute_spo2_from_rppg(self):
        if len(self.rppg_buffer) < 6:
            return
        reds = np.array([r for (_, g, r) in self.rppg_buffer])
        greens = np.array([g for (_, g, r) in self.rppg_buffer])
        ac_r = np.std(reds)
        dc_r = np.mean(reds) + 1e-6
        ac_g = np.std(greens)
        dc_g = np.mean(greens) + 1e-6
        rr = (ac_r / dc_r) / (ac_g / dc_g + 1e-6)
        spo2 = 100 - (rr - 0.5) * 10.0
        spo2 = max(80.0, min(100.0, spo2 * SP02_HEURISTIC_FACTOR))
        self.spO2 = 0.75 * self.spO2 + 0.25 * spo2

    def compute_skin_temp_from_face(self):
        if len(self.rppg_buffer) < 3:
            return
        latest_g = self.rppg_buffer[-1][1]
        delta = ((latest_g / 255.0) - 0.5) * 1.6
        temp = TEMP_BASELINE + delta
        self.skin_temp = 0.9 * self.skin_temp + 0.1 * temp

    # Exercise detection helpers
    def _slug(self, name):
        return name.lower().replace(" ", "_")

    def _avg_angle(self, key, value):
        buf = self.angle_buffers[key]
        buf.append(value)
        return float(sum(buf) / len(buf))

    # Specific detectors (use left landmarks primarily with symmetry fallback)
    def _detect_straight_posture(self, landmarks, img_shape, ex):
        try:
            Ls = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], img_shape)
            Lh = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value], img_shape)
            Lk = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value], img_shape)
        except Exception:
            return "No landmarks", None, "L"
        angle = self._avg_angle(ex, calculate_angle(Ls, Lh, Lk))
        min_a, max_a = EXERCISE_CONFIG[ex]["angle_range"]
        feedback = "Good Posture!" if min_a <= angle <= max_a else "Keep back straight!"
        self.update_fitness_factors(ex)
        return feedback, angle, "L"

    def _detect_head_rotation(self, landmarks, img_shape, ex):
        try:
            Le = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value], img_shape)
            Re = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value], img_shape)
        except Exception:
            return "No landmarks", None, "C"
        diff = abs(Le[0]-Re[0]) / img_shape[1] if img_shape[1] > 0 else 0
        threshold = EXERCISE_CONFIG[ex]["turn_threshold"]
        if diff > threshold and self.stages[ex] != "turned":
            self.counters[ex] += 1
            self.stages[ex] = "turned"
            self._speak_quiet(f"Head rotation {self.counters[ex]}")
            feedback = f"Turned ({diff:.2f})"
            self.update_fitness_factors(ex)
        else:
            if diff <= threshold:
                self.stages[ex] = "center"
            feedback = f"Center ({diff:.2f})" if diff <= threshold else f"Turned ({diff:.2f})"
        return feedback, diff*100.0, "C"

    def _detect_jumping_jacks(self, landmarks, img_shape, ex):
        try:
            Ls = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], img_shape)
            Lh = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value], img_shape)
            La = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value], img_shape)
        except Exception:
            return "No landmarks", None, "L"
        angle = self._avg_angle(ex, calculate_angle(Ls, Lh, La))
        cfg = EXERCISE_CONFIG[ex]
        if angle > cfg["min_open"]:
            self.stages[ex] = "open"
            feedback = "Open"
        elif angle < cfg["min_close"] and self.stages[ex] == "open":
            self.counters[ex] += 1
            self.stages[ex] = "closed"
            self._speak_quiet(f"Jumping Jack {self.counters[ex]}")
            feedback = "Good Rep"
            self.update_fitness_factors(ex)
        else:
            feedback = "Moving"
        return feedback, angle, "L"

    def _generic_arm_ex(self, name, landmarks, img_shape, ex):
        try:
            s = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], img_shape)
            e = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value], img_shape)
            w = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value], img_shape)
        except Exception:
            return "No landmarks", None, "L"
        angle = self._avg_angle(ex, calculate_angle(s, e, w))
        cfg = EXERCISE_CONFIG[name]
        feedback = "Moving"
        if name == "Push-ups":
            if angle > cfg["up_min"]:
                self.stages[ex] = "up"; feedback = "Up"
            elif angle < cfg["down_max"] and self.stages[ex] == "up":
                self._rep(ex, f"Push-up {self.counters[ex]}")
                feedback = "Good Push-up"
        elif name == "Pull-ups":
            if angle < cfg["up_max"]:
                self.stages[ex] = "up"; feedback = "Pulled Up"
            elif angle > cfg["down_min"] and self.stages[ex] == "up":
                self._rep(ex, f"Pull-up {self.counters[ex]}")
                feedback = "Good Pull-up"
        elif name == "Bicep Curls":
            if angle > cfg["down_min"]:
                self.stages[ex] = "down"; feedback = "Arm Down"
            elif angle < cfg["up_max"] and self.stages[ex] == "down":
                self._rep(ex, f"Bicep Curl {self.counters[ex]}")
                feedback = "Good Curl"
        return feedback, angle, "L"

    def _generic_leg_ex(self, name, landmarks, img_shape, ex):
        try:
            hpt = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value], img_shape)
            kpt = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value], img_shape)
            apt = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value], img_shape)
        except Exception:
            return "No landmarks", None, "L"
        angle = self._avg_angle(ex, calculate_angle(hpt, kpt, apt))
        cfg = EXERCISE_CONFIG[name]
        feedback = "Moving"
        if name in ["Squats", "Lunges", "Leg Raise", "Burpees"]:
            up_min = cfg.get("up_min", cfg.get("stand_min", 160))
            down_max = cfg.get("down_max", 90)
            if angle > up_min:
                self.stages[ex] = "up"; feedback = "Up"
            elif angle < down_max and self.stages[ex] == "up":
                self._rep(ex, f"{name} {self.counters[ex]}")
                feedback = f"Good {name}"
        return feedback, angle, "L"

    def _generic_plank(self, landmarks, img_shape, ex):
        try:
            s = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], img_shape)
            hpt = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value], img_shape)
            a = to_pixel_coords(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value], img_shape)
        except Exception:
            return "No landmarks", None, "L"
        angle = self._avg_angle(ex, calculate_angle(s, hpt, a))
        min_a, max_a = EXERCISE_CONFIG["Plank"]["angle_range"]
        feedback = "Perfect Plank!" if min_a <= angle <= max_a else "Keep hips level!"
        return feedback, angle, "L"

    def _detect_walking(self, landmarks, img_shape, ex):
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_y = (left_hip.y + right_hip.y) / 2.0
        h, w = img_shape[:2]
        if self.prev_hip_y is None:
            self.prev_hip_y = hip_y
            return "Walking init", 0.0, "C"
        diff = abs(hip_y - self.prev_hip_y)
        self.prev_hip_y = hip_y
        cfg = EXERCISE_CONFIG[ex]
        if diff > cfg.get("step_mag", 0.02):
            self.counters[ex] += 1
            self._speak_quiet(f"Step {self.counters[ex]}")
            self.update_fitness_factors(ex)
            return "Step detected", diff * 100.0, "C"
        return "Walking", diff * 100.0, "C"

    def _detect_calf_raises(self, landmarks, img_shape, ex):
        ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        h, w = img_shape[:2]
        ankle_y = ankle.y
        cfg = EXERCISE_CONFIG[ex]
        if self.prev_ankle_y is None:
            self.prev_ankle_y = ankle_y
            self.stages[ex] = "down"
            return "Calf init", 0.0, "L"
        diff = (self.prev_ankle_y - ankle_y) * h
        self.prev_ankle_y = ankle_y
        if diff > cfg.get("up_min", 10) and self.stages[ex] != "up":
            self.stages[ex] = "up"
            return "Up on toes", diff, "L"
        elif diff < -cfg.get("down_max", 5) and self.stages[ex] == "up":
            self._rep(ex, f"Calf Raise {self.counters[ex]}")
            return "Good Calf Raise", diff, "L"
        return "Calf moving", diff, "L"

    def _rep(self, ex, text):
        self.stages[ex] = "down"
        self.counters[ex] += 1
        self._speak_quiet(text)
        self.update_fitness_factors(ex)

    def _speak_quiet(self, text):
        now = time.time()
        if now - self.last_speech_time > self.speech_cooldown:
            self.tts.speak(text)
            self.last_speech_time = now

    def reset_exercise(self, ex):
        self.counters[ex] = 0
        self.stages[ex] = None
        self.angle_buffers[ex].clear()
        self.rppg_buffer.clear()
        self.breath_buffer.clear()
        self.hr_buffer.clear()
        self.hr_time_series.clear()
        self.start_time = time.time()
        self.last_hydration = time.time()
        self.prev_hip_y = None
        self.prev_ankle_y = None

    def log_session(self):
        row = {
            "timestamp": datetime.now().isoformat(),
            "counts": {ex: self.counters[ex] for ex in EXERCISE_LIST},
            "duration_sec": int(self.total_time),
            "calories_burned": round(self.calories_burned, 2),
            "heart_rate": int(self.heart_rate),
            "breath_rate": round(self.breath_rate, 1),
            "spO2": round(self.spO2, 1),
            "skin_temp": round(self.skin_temp, 2),
            "bp_systolic": self.bp_systolic,
            "bp_diastolic": self.bp_diastolic,
            "intensity": self.intensity,
            "bmi": round(self.bmi, 2),
            "fatigue_index": round(self.fatigue_index, 1),
            "stress_index": round(self.stress_index, 1)
        }
        self.session_log.append(row)
        return row

    def save_log_csv(self, filename=None):
        filename = filename or f"session_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp", "duration_sec", "calories_burned", "heart_rate", "breath_rate",
                      "spO2", "skin_temp", "bp_systolic", "bp_diastolic", "intensity", "bmi",
                      "fatigue_index", "stress_index"] + EXERCISE_LIST
            writer.writerow(header)
            for e in self.session_log:
                row = [e.get("timestamp"), e.get("duration_sec"), e.get("calories_burned"), e.get("heart_rate"),
                       e.get("breath_rate"), e.get("spO2"), e.get("skin_temp"), e.get("bp_systolic"),
                       e.get("bp_diastolic"), e.get("intensity"), e.get("bmi"), e.get("fatigue_index"),
                       e.get("stress_index")] + [e["counts"].get(ex, 0) for ex in EXERCISE_LIST]
                writer.writerow(row)
        return filename

    # Main processing
    def process(self, image, selected_exercise):
        if self.start_time is None:
            self.start_time = time.time()
        self.total_time = time.time() - self.start_time

        # rPPG sampling
        self.compute_rppg_from_face(image)

        # pose detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(image_rgb)
        annotated = image.copy()
        h, w = image.shape[:2]

        feedback = ""
        angle_value = None
        if results_pose.pose_landmarks:
            self.mp_drawing.draw_landmarks(annotated, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = results_pose.pose_landmarks.landmark
            self.append_breath_sample(landmarks, image.shape)

            method = getattr(self, f"_detect_{self._slug(selected_exercise)}", None)
            if method is None:
                # map to generic detectors
                if selected_exercise in ["Push-ups", "Pull-ups", "Bicep Curls"]:
                    method = lambda l, s, e: self._generic_arm_ex(selected_exercise, l, s, e)
                elif selected_exercise in ["Squats", "Lunges", "Leg Raise", "Burpees"]:
                    method = lambda l, s, e: self._generic_leg_ex(selected_exercise, l, s, e)
                elif selected_exercise in ["Plank"]:
                    method = lambda l, s, e: self._generic_plank(l, s, e)
                else:
                    method = None
            if method:
                try:
                    feedback, angle_value, _ = method(landmarks, image.shape, selected_exercise)
                except Exception:
                    feedback = "Detect error"

            if angle_value is not None:
                try:
                    cv2.putText(annotated, f"Angle: {angle_value:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 2)
                except Exception:
                    pass

            cv2.putText(annotated, f"{selected_exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(annotated, f"Count: {self.counters.get(selected_exercise,0)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if feedback:
                color = (0,200,0)
                if any(x in feedback.lower() for x in ["keep","bad","wrong","error","too"]):
                    color = (0,0,200)
                cv2.putText(annotated, feedback, (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # vitals compute
        self.compute_hr_from_rppg()
        self.compute_breath_rate()
        self.compute_spo2_from_rppg()
        self.compute_skin_temp_from_face()
        self.update_fitness_factors(selected_exercise)

        # overlay vitals and metrics (right side)
        x0 = max(10, int(w * 0.6))
        cv2.putText(annotated, f"Time: {int(self.total_time)}s", (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(annotated, f"Calories: {self.calories_burned:.2f} kcal", (x0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 2)
        cv2.putText(annotated, f"HR: {int(self.heart_rate)} BPM", (x0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,100,100), 2)
        cv2.putText(annotated, f"BR: {self.breath_rate:.1f} bpm", (x0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2)
        cv2.putText(annotated, f"SpO2: {self.spO2:.0f} %", (x0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 2)
        cv2.putText(annotated, f"SkinT: {self.skin_temp:.1f} C", (x0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
        cv2.putText(annotated, f"BP: {self.bp_systolic}/{self.bp_diastolic} mmHg", (x0, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,180), 2)
        cv2.putText(annotated, f"Intensity: {self.intensity}", (x0, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(annotated, f"BMI: {self.bmi:.1f}", (x0, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)
        cv2.putText(annotated, f"Fatigue: {int(self.fatigue_index)}", (x0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,180,180), 2)
        cv2.putText(annotated, f"Stress: {int(self.stress_index)}", (x0, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,200), 2)

        if self.hydration_reminder_flag:
            cv2.putText(annotated, "Hydration: Take a sip!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 3)

        return annotated

# --------------------- Main App ---------------------
def main():
    print("\n=== AI FITNESS TRAINER (Medical PRO) ===\n")
    # personalization inputs
    try:
        weight = float(input("Enter your weight in kg (or press Enter for 70): ") or 70)
    except Exception:
        weight = 70.0
    try:
        height_cm = float(input("Enter your height in cm (or press Enter for 175): ") or 175)
        height_m = height_cm / 100.0
    except Exception:
        height_m = 1.75
    try:
        age = int(input("Enter your age (or press Enter for 30): ") or 30)
    except Exception:
        age = 30

    tts = TTSWorker(enabled=True)  # default enabled; toggle with 's'
    detector = ExerciseDetector(tts_worker=tts, weight_kg=weight, height_m=height_m, age=age)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not detected. Run as script with a connected webcam.")
        return

    # choose exercise
    print("\nAvailable exercises:")
    for i, ex in enumerate(EXERCISE_LIST, 1):
        print(f"{i}. {ex}")
    print("Type exercise name or number. Press Enter to start with Straight Posture.")
    choice = input("Enter exercise name or number: ").strip()
    if choice == "":
        selected_ex = "Straight Posture"
    elif choice.isdigit():
        idx = int(choice)-1
        selected_ex = EXERCISE_LIST[idx] if 0 <= idx < len(EXERCISE_LIST) else "Straight Posture"
    else:
        selected_ex = None
        name_try = choice.strip().title()
        for ex in EXERCISE_LIST:
            if ex.lower() == choice.strip().lower() or ex == name_try:
                selected_ex = ex
                break
        if selected_ex is None:
            print("Unknown exercise, defaulting to Straight Posture")
            selected_ex = "Straight Posture"

    print(f"\n✅ Starting with: {selected_ex}")
    video_filename = next_video_filename(prefix="video", ext="avi")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame_width, frame_height))

    print("Controls: q=quit & save | e=next exercise | s=toggle speech | r=reset counters | l=save logs now | h=reset hydration")
    fps_buf = deque(maxlen=FPS_SMOOTHING)
    last = time.time()
    selected_idx = EXERCISE_LIST.index(selected_ex)
    detector.start_time = time.time()

    # live loop
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Camera read failed.")
            break
        frame = cv2.flip(frame, 1)

        annotated = detector.process(frame, selected_ex)

        # FPS calc
        now = time.time()
        fps = 1.0 / (now - last) if now - last > 0 else 0.0
        last = now
        fps_buf.append(fps)
        avg_fps = sum(fps_buf) / len(fps_buf) if len(fps_buf) > 0 else 0.0
        cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, annotated.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(annotated, f"Exercise ({selected_idx+1}/{len(EXERCISE_LIST)}): {selected_ex}", (10, annotated.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

        # write file
        out.write(annotated)

        # show
        cv2.imshow("AI Fitness Trainer - Medical PRO", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting...")
            tts.speak("Workout session ended. Good job!")
            break
        elif key == ord('e'):
            selected_idx = (selected_idx + 1) % len(EXERCISE_LIST)
            selected_ex = EXERCISE_LIST[selected_idx]
            print("Switched to:", selected_ex)
            tts.speak(f"Switched to {selected_ex}")
            # reset timers for the new exercise
            detector.start_time = time.time()
        elif key == ord('r'):
            detector.reset_exercise(selected_ex)
            print(f"Reset counters for {selected_ex}")
            tts.speak("Reset counters")
        elif key == ord('s'):
            enabled = tts.toggle()
            print("Speech enabled:", enabled)
        elif key == ord('l'):
            row = detector.log_session()
            saved = detector.save_log_csv()
            print("Logged & saved session to", saved)
            tts.speak("Session logged")
        elif key == ord('h'):
            detector.last_hydration = time.time()
            detector.hydration_reminder_flag = False
            print("Hydration reminder reset")
            tts.speak("Hydration timer reset")
    # cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    tts.stop()

    # final save & summary
    try:
        final_row = detector.log_session()
        final_file = detector.save_log_csv()
        print("Final session saved to", final_file)
    except Exception as e:
        print("Could not save final log:", e)

    print("\n===== 🏁 WORKOUT SUMMARY =====")
    print(f"Duration: {int(detector.total_time)} sec")
    print(f"Calories Burned: {detector.calories_burned:.2f} kcal")
    print(f"Heart Rate (approx): {detector.heart_rate:.1f} BPM")
    print(f"Breathing Rate (approx): {detector.breath_rate:.1f} bpm")
    print(f"SpO2 (approx): {detector.spO2:.1f} %")
    print(f"Skin Temp (approx): {detector.skin_temp:.2f} °C")
    print(f"Blood Pressure (est): {detector.bp_systolic}/{detector.bp_diastolic} mmHg")
    print(f"Intensity Level: {detector.intensity}")
    print(f"BMI: {detector.bmi:.2f}")
    print(f"Fatigue Index: {detector.fatigue_index:.1f}")
    print(f"Stress Index: {detector.stress_index:.1f}")
    print("Reps Count:")
    for ex in EXERCISE_LIST:
        print(f" - {ex}: {detector.counters[ex]}")
    print("==============================\n")

if __name__ == "__main__":
    main()