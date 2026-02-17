from ultralytics import YOLO

# Load models once at startup (efficient)
model_v8 = YOLO("models/best.pt")
model_v9 = YOLO("models/v9_best.pt")

def load_model(version: str):
    if version == "v8":
        return model_v8
    elif version == "v9":
        return model_v9
    else:
        raise ValueError("Invalid model version")
