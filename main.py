from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
import cv2
from math import sqrt

app = FastAPI()

class MeasureRequest(BaseModel):
    image_url: str
    garment_type: str = "trouser"

class TapRequest(BaseModel):
    image_url: str
    points: list

def download_image(url):
    resp = requests.get(url)
    img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def detect_card_width(img):
    # dummy detection (later improve)
    h, w, _ = img.shape
    return w * 0.15

def pixel_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

@app.post("/measure")
def measure(data: MeasureRequest):
    img = download_image(data.image_url)

    card_width_px = detect_card_width(img)

    if card_width_px == 0:
        return {"needs_taps": True}

    pixel_to_mm = 85.6 / card_width_px

    # Dummy waist detection (replace with real edge detection later)
    waist_px = img.shape[1] * 0.5

    waist_mm = waist_px * pixel_to_mm
    waist_inch = waist_mm / 25.4

    size = round(waist_inch)

    return {
        "needs_taps": False,
        "measurements": {
            "waist": round(waist_inch,1),
            "length": 40
        },
        "best_match": str(size),
        "confidence": "Medium",
        "alternatives": [str(size-1), str(size+1)]
    }

@app.post("/measure-with-taps")
def measure_with_taps(data: TapRequest):
    img = download_image(data.image_url)

    p1, p2, p3, p4 = data.points[:4]

    waist_px = pixel_distance(p1, p2)

    card_px = img.shape[1] * 0.15
    pixel_to_mm = 85.6 / card_px

    waist_inch = (waist_px * pixel_to_mm) / 25.4

    size = round(waist_inch)

    return {
        "needs_taps": False,
        "measurements": {
            "waist": round(waist_inch,1),
            "length": 40
        },
        "best_match": str(size),
        "confidence": "High",
        "alternatives": [str(size-1), str(size+1)]
    }
