from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from evals.utils import load_embedding_model, evaluate_all
from typing import List, Dict
import os
import json

models = {}

UPLOAD_DIR = "uploads"
METADATA_FILE = os.path.join(UPLOAD_DIR, "metadata.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_metadata() -> dict:
    """Load metadata from JSON file, or return defaults if it doesn't exist or is empty/corrupt."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {"upload_counter": 0, "files": {}}
                return json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ metadata.json is corrupt, resetting to defaults.")
            return {"upload_counter": 0, "files": {}}
    return {"upload_counter": 0, "files": {}}


def save_metadata(metadata: dict):
    """Persist metadata to JSON file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# Load on module init
metadata = load_metadata()
uploads: Dict[str, Dict] = metadata["files"]
upload_counter: int = metadata["upload_counter"]

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
    for filename in os.listdir(UPLOAD_DIR):
        if filename == "metadata.json":
            continue  # skip the metadata file itself
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(path):
            upload_id = uploads.get(filename, {}).get("upload_id")
            files.append({
                "upload_id": upload_id,
                "filename": filename,
                "size": os.path.getsize(path)
            })
    return {"files": files}



@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    global upload_counter
    results = []
    for file in files:

        upload_counter += 1
        upload_id = upload_counter
        contents = (await file.read()).decode("utf-8", errors="ignore")
        uploads[file.filename] = {
            "content": contents,
            "upload_id": upload_id
        }

        with open(f"uploads/{file.filename}", "wb") as f:
            f.write(contents.encode("utf-8"))

        results.append({
            "upload_id": upload_id,
            "filename": file.filename,
            "content": contents,
        })
        
    save_metadata({
        "upload_counter": upload_counter,
        "files": uploads
    })
    return {"files": results}



@app.get("/evaluations/")
async def get_evaluations():
    """Return cached evaluations from metadata without re-running."""
    results = {}
    for filename, file_data in uploads.items():
        if "evaluation" in file_data:
            results[filename] = file_data["evaluation"]
    return {"evaluations": results}




@app.get("/evaluate/")
async def evaluate():
    results = {}

    for filename, file_data in uploads.items():
        content = file_data.get("content", "")
        if not content:
            continue

        try:
            eval_result = evaluate_all(content)

            # Convert non-serializable types before saving
            eval_summary = {
                "total_words": eval_result["total_words"],
                "unique_words": eval_result["unique_words"],
                "diversity_compression_ratio": eval_result["diversity_compression_ratio"],
                "ngram_diversity": eval_result["ngram_diversity"],
                "dpp_log_determinant": float(eval_result["dpp_log_determinant"]),
                "dpp_sign": int(eval_result["dpp_sign"]),
                "diversity_percentage": float(eval_result["diversity_percentage"]),
                "vendi_score": float(eval_result["vendi_score"]),
                "novascore": float(eval_result["novascore"][1]) if eval_result["novascore"] else 0.0,
            }

            # Append evaluation to the file's metadata entry
            uploads[filename]["evaluation"] = eval_summary
            results[filename] = eval_summary

        except Exception as e:
            results[filename] = {"error": str(e)}

    # Persist updated metadata
    save_metadata({
        "upload_counter": upload_counter,
        "files": uploads
    })

    return {"evaluations": results}