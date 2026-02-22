import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.predict import predict_audio

# Use temporary directory (Render-safe)
UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded file temporarily
    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = predict_audio(file_path)

    # Optional: delete file after prediction
    os.remove(file_path)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result}
    )


@app.get("/health")
def health():
    return {"status": "ok"}