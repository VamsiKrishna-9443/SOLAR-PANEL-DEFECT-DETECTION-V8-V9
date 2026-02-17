from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.predictor import predict_image

app = FastAPI(title="Solar AI Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results folder (AFTER app creation)
app.mount(
    "/results",
    StaticFiles(directory="runs/detect/results"),
    name="results"
)




@app.get("/")
def home():
    return {"message": "Solar AI Backend Running"}

@app.post("/predict/yolov8")
def predict_yolov8(file: UploadFile = File(...)):
    return predict_image(file, "v8")

@app.post("/predict/yolov9")
def predict_yolov9(file: UploadFile = File(...)):
    return predict_image(file, "v9")
