from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn
from utils import process_company_news
import os
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Sentiment Analysis API",
    description="API for extracting, analyzing, and converting news articles to speech",
    version="1.0.0"
)

class CompanyRequest(BaseModel):
    company_name: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the News Sentiment Analysis API"}

@app.post("/analyze")
async def analyze_company(request: CompanyRequest):
    """
    Analyze news articles for a given company.
    
    Returns sentiment analysis, comparative analysis, and generates Hindi TTS.
    """
    try:
        logger.info(f"Processing request for company: {request.company_name}")
        result = process_company_news(request.company_name)
        
        # Modify response to handle the audio file path
        audio_path = result.get("Audio")
        if audio_path:
            result["Audio"] = f"/get_audio?company={request.company_name}"
        
        return result
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Return a simplified response with error information
        return JSONResponse(
            status_code=200,  # Return 200 instead of 500 to show error in UI
            content={
                "Company": request.company_name,
                "Articles": [{
                    "Title": "Error retrieving news",
                    "Summary": f"An error occurred: {str(e)}. Please try again later.",
                    "Sentiment": "Neutral",
                    "Topics": ["error"]
                }],
                "Comparative Sentiment Score": {
                    "Sentiment Distribution": {"Positive": 0, "Neutral": 1, "Negative": 0},
                    "Coverage Differences": [],
                    "Topic Overlap": {"Common Topics": [], "All Topics": ["error"]}
                },
                "Final Sentiment Analysis": "Error occurred during analysis.",
                "Hindi Summary": "त्रुटि हुई है। बाद में पुन: प्रयास करें।",
                "Audio": None
            }
        )

@app.get("/get_audio")
async def get_audio(company: str = Query(None)):
    """
    Retrieve the generated audio file for a company.
    """
    try:
        # This is a simplified approach. In production, you'd want to store
        # audio files with unique IDs in a database or file system
        result = process_company_news(company)
        audio_path = result.get("Audio")
        
        if audio_path and os.path.exists(audio_path):
            return FileResponse(
                audio_path, 
                media_type="audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav",
                filename=f"{company}_sentiment.mp3" if audio_path.endswith(".mp3") else f"{company}_sentiment.wav"
            )
        else:
            raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        logger.error(f"Error retrieving audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving audio: {str(e)}")

@app.get("/companies")
async def get_sample_companies():
    """
    Get a list of sample companies for the dropdown.
    """
    return {
        "companies": [
            "Tesla", "Apple", "Microsoft", "Amazon", "Google", 
            "Meta", "Netflix", "Nvidia", "Intel", "AMD"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)