from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from evals.utils import load_embedding_model
from typing import List
import os

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["embedding"] = load_embedding_model()
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)



@app.get("/files/")
async def list_files():
    files = []
    for filename in os.listdir("uploads"):
        path = os.path.join("uploads", filename)
        if os.path.isfile(path):
            files.append({
                "filename": filename,
                "size": os.path.getsize(path)
            })
    return {"files": files}



@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        with open(f"uploads/{file.filename}", "wb") as f:
            f.write(contents)
        results.append({
            "filename": file.filename,
            "status": "success",
            "size": len(contents)
        })
    return {"files": results}



@app.get("/evaluate/")
async def evaluate():
    pass