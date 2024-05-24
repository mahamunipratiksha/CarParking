from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

from pydantic import BaseModel
import parking_slot_detection


class VideoPredParams(BaseModel):
    video: UploadFile
    coordinates: UploadFile
    time_interval: int = 1

app = FastAPI()


# # ... (other imports and code)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","http://localhost:5173"],  # Update this to your specific frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="results"), name="static")


import os


@app.post("/get_video")
async def get_video(video_data: dict):
    video_path = video_data.get("video_path")
    if not video_path:
        raise HTTPException(status_code=422, detail="Video path is required")
    # Assuming the video files are stored in the 'result' directory
    print("Received video path:", video_path)
    video_file_path = f"./results{video_path}"
    
    # Return the video file as a response
    return FileResponse(video_file_path, media_type="video/mp4")


@app.post("/video_prediction/")
async def get_data(video: UploadFile = File(...),coordinates: UploadFile = File(...)):
    
    try:
        predict_interval=1
        # Save the video and coordinates files locally
        video_path = f"uploaded_files/{video.filename}"  # Update the path as needed
        coordinates_path = f"uploaded_files/{coordinates.filename}"  # Update the path as needed

        print("received in backend ",video.filename)
        with open(video_path, "wb") as video_file:
            video_file.write(await video.read())

        with open(coordinates_path, "wb") as coordinates_file:
            coordinates_file.write(await coordinates.read())

        print("video path : ",video_path)
        print("coordinates path : ",coordinates_path)
        output_vid_path,output_pred_path = parking_slot_detection.get_vid_predictions(video_path,coordinates_path,predict_interval)
        
        d = {}
        d['pred_vid_path'] = output_vid_path
        d['pred_file_path'] = output_pred_path
        return d

    except Exception as e:
        print(f"Error in retrieving and processing data: {e}")
        return {}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info",reload=True)