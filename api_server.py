from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2

from ai_fitness_trainer_medical_pro import ExerciseDetector, TTSWorker

app = FastAPI(title="AI Fitness Trainer ML API")

tts = TTSWorker(enabled=False)
detector = None  

@app.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    exercise: str = Form("Squats"),
    weight: float = Form(70),
    height_cm: float = Form(175),
    age: int = Form(30),
):
    global detector

    # Create detector once
    if detector is None:
        detector = ExerciseDetector(
            tts_worker=tts,
            weight_kg=weight,
            height_m=height_cm / 100,
            age=age
        )
        detector.current_exercise = exercise

    # Reset if exercise changes
    if detector.current_exercise != exercise:
     detector.reset_exercise(exercise)
     detector.current_exercise = exercise


    # Decode frame
    image_bytes = await file.read()
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid frame"}

    # Run AI
    detector.process(frame, exercise)

    # Return unified live metrics
    return detector.get_live_metrics(exercise)


@app.post("/reset")
def reset_exercise(exercise: str = Form("Squats")):
    if detector:
        detector.reset_exercise(exercise)
    return {"status": "reset", "exercise": exercise}
