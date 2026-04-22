from fastapi import FastAPI, HTTPException
from schemas import FormTrendRequest, FormTrendResponse
from services import compute_form_trend

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/api/v1/player/form-trend", response_model=FormTrendResponse)
def get_form_trend(request: FormTrendRequest):
    try:
        return compute_form_trend(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
