from infer_llama2 import infer_
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

class Query(BaseModel):
    query: str


app = FastAPI()


@app.get("/health_check")
def get_health():
    return {"message": "Hello World"}
    
@app.post("/tesing")
def infer_llama_ct2( q: Query):
    try:
        resp, tps = infer_(q.query)
        return {
            "response": resp,
            "tps": tps
        }
    except:
        return {
            "response": "ERROR"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", port=1521, host = "0.0.0.0")