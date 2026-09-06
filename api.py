from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from train import InferenceService

inference_service = InferenceService()
app = FastAPI()


class Text(BaseModel):
    text: str

class Response(BaseModel):
    label: str
    confidence: float


@app.post("/predict", response_model=Response)
async def predict(request: Text):
    try:
        result = inference_service.predict(request.text)
        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "device": inference_service.device}
