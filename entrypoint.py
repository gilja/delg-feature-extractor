from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing_extensions import Literal
import numpy as np
from io import BytesIO
from PIL import Image

from delg import extractor
from delg.utils import _load_config, _default_config_path
from typing import Dict

app = FastAPI()


@app.on_event("startup")
def load_models():
    """Load the DELG extractors once at startup and store in app state."""
    app.state.extractors = {}
    for mode in ["global", "local"]:
        try:
            config_path = _default_config_path(mode)
            config = _load_config(config_path)
            app.state.extractors[mode] = extractor.MakeExtractor(config)
            print(f"✅ Loaded DELG {mode} extractor.")
        except Exception as e:
            print(f"❌ Failed to load {mode} extractor: {e}")
            raise


def extract_features(image_bytes: bytes, mode: Literal["global", "local"]) -> Dict:
    """Run DELG extraction on raw image bytes."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_np = np.array(image)
    extractor_fn = app.state.extractors[mode]
    features = extractor_fn(image_np)

    if mode == "global":
        vector = features.get("global_descriptor")
        if vector is None:
            raise HTTPException(
                status_code=500, detail="Failed to compute global features"
            )
        return {"global_descriptor": vector.tolist()}

    local = features.get("local_features")
    if local is None:
        raise HTTPException(status_code=500, detail="Failed to compute local features")
    return {
        "local_features": {
            "locations": local["locations"].tolist(),
            "descriptors": local["descriptors"].tolist(),
            "scales": local["scales"].tolist(),
            "attention": local["attention"].tolist(),
        }
    }


@app.post("/extract/global")
async def extract_global(image: UploadFile = File(...)):
    """POST endpoint to extract global features from an image."""
    image_bytes = await image.read()
    result = extract_features(image_bytes, mode="global")
    return JSONResponse(content=result)


@app.post("/extract/local")
async def extract_local(image: UploadFile = File(...)):
    """POST endpoint to extract local features from an image."""
    image_bytes = await image.read()
    result = extract_features(image_bytes, mode="local")
    return JSONResponse(content=result)
