import os
import shutil
from app.model_loader import load_model

BASE_URL = "http://127.0.0.1:8000"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def predict_image(file, model_version):

    model = load_model(model_version)

    # 1️⃣ Save uploaded file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2️⃣ Run YOLO inference
    results = model(file_path, save=True, project=".", name="results", exist_ok=True)

    # 3️⃣ Get save directory (predictX)
    save_dir = results[0].save_dir
    folder = os.path.basename(save_dir)

    # 4️⃣ Get predicted image filename
    predicted_files = [
        f for f in os.listdir(save_dir)
        if f.endswith((".jpg", ".png"))
    ]
    predicted_filename = predicted_files[-1]

    image_url = f"http://127.0.0.1:8000/results/{predicted_filename}"

    return {
    "image_url": f"http://127.0.0.1:8000/results/{predicted_filename}",
    "model_used": model_version,
    "status": "success"
}
