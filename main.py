import os
import uvicorn
import logging
from body_scan import scan_the_body
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(openapi_url="/openapi.json", docs_url="/docs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def find_closest_shirt_size(chest, length):
    shirt_size_chart = {
        "XS": {"chest": 86.36, "length": 63.5},
        "S": {"chest": 93.98, "length": 66.04},
        "M": {"chest": 106.68, "length": 68.58},
        "L": {"chest": 109.22, "length": 71.12},
        "XL": {"chest": 116.84, "length": 73.66},
        "XXL": {"chest": 124.46, "length": 76.2}
    }

    closest_size = min(
        shirt_size_chart.keys(),
        key=lambda size: abs(
            shirt_size_chart[size]["chest"] - chest) + abs(shirt_size_chart[size]["length"] - length)
    )

    return closest_size


def find_closest_trouser_size(waist, length):
    trouser_size_chart = {
        "XS": {"waist": 73.66, "length": 101.6},
        "S": {"waist": 78.74, "length": 104.14},
        "M": {"waist": 83.82, "length": 106.68},
        "L": {"waist": 88.9, "length": 109.22},
        "XL": {"waist": 93.98, "length": 111.76},
        "XXL": {"waist": 99.06, "length": 114.3}
    }

    closest_size = min(
        trouser_size_chart.keys(),
        key=lambda size: abs(trouser_size_chart[size]["waist"] - waist) + abs(
            trouser_size_chart[size]["length"] - length)
    )

    return closest_size


TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    temp_file_path = os.path.join(TEMP_DIR, file.filename)

    try:
        # Save the uploaded file to the temporary location using aiofiles
        async with aiofiles.open(temp_file_path, 'wb') as temp_file:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=400, detail="Uploaded file is empty")
            await temp_file.write(content)

        logger.info(f"File saved: {temp_file_path}")

        try:
            measurements = scan_the_body(temp_file_path)
            logger.info(f"Scan results: {measurements}")

            if len(measurements) != 5:
                raise ValueError(f"Expected 5 measurements from scan_the_body, got {len(measurements)}")

            waist, shoulder, arm, leg, chest = measurements

            logger.info(f"Scanned Measurements: Waist={waist}, Shoulder={shoulder}, Arm={arm}, Leg={leg}, Chest={chest}")

            shirt_size = find_closest_shirt_size(chest, arm)
            trouser_size = find_closest_trouser_size(waist, leg)

            logger.info(f"Determined Sizes: Shirt={shirt_size}, Trouser={trouser_size}")

            return {"shirt": shirt_size, "trouser": trouser_size}

        except Exception as e:
            logger.error(
                f"Error in scan_the_body or size determination: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error processing image: {str(e)}")

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file removed: {temp_file_path}")

if __name__ == '__main__':
    uvicorn.run(app, port=8000)
