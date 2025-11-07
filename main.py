from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Body
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from services.storage import CloudStorageService
from services.ai_generation import PhotoBoothGenerator
import uuid
from fastapi import Form
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import time

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Photobooth", description="Generate AI photo booth images from selfies")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
# Only mount static files if directory exists
import os
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
# Initialize services (delay AI generator initialization)
storage_service = CloudStorageService()
ai_generator = None

# Initialize services
storage_service = CloudStorageService()

# Configuration for image cleanup
IMAGE_RETENTION_MINUTES = int(os.getenv('IMAGE_RETENTION_MINUTES', '2'))  # Default: 24 hours
CLEANUP_INTERVAL_MINUTES = int(os.getenv('CLEANUP_INTERVAL_MINUTES', '2'))  # Default: 60 minutes
TEMP_IMAGES_DIR = "temp_images"

# Ensure temp directory exists
os.makedirs(TEMP_IMAGES_DIR, exist_ok=True)

def get_ai_generator():
    """Get AI generator instance with proper error handling"""
    try:
        from services.ai_generation import PhotoBoothGenerator
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        generator = PhotoBoothGenerator()
        print(f"AI generator initialized successfully")
        return generator
    except Exception as e:
        print(f"Error initializing AI generator: {e}")
        raise HTTPException(status_code=500, detail=f"AI service unavailable: {str(e)}")

# Store generation jobs (in production, use a database)
generation_jobs = {}

async def cleanup_old_images():
    """
    Periodically delete images older than IMAGE_RETENTION_MINUTES
    """
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL_MINUTES * 60)  # Convert minutes to seconds
            
            print(f"[CLEANUP] Starting cleanup task...")
            current_time = time.time()
            retention_seconds = IMAGE_RETENTION_MINUTES * 60
            deleted_count = 0
            
            # Check if temp directory exists
            if not os.path.exists(TEMP_IMAGES_DIR):
                print(f"[CLEANUP] Directory {TEMP_IMAGES_DIR} does not exist")
                continue
            
            # Iterate through all files in temp_images directory
            for filename in os.listdir(TEMP_IMAGES_DIR):
                file_path = os.path.join(TEMP_IMAGES_DIR, filename)
                
                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue
                
                # Get file modification time
                file_mtime = os.path.getmtime(file_path)
                file_age_seconds = current_time - file_mtime
                
                # Delete if older than retention period
                if file_age_seconds > retention_seconds:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        file_age_hours = file_age_seconds / 3600
                        print(f"[CLEANUP] Deleted old image: {filename} (age: {file_age_hours:.2f} hours)")
                    except Exception as e:
                        print(f"[CLEANUP] Error deleting {filename}: {e}")
            
            if deleted_count > 0:
                print(f"[CLEANUP] Cleanup complete. Deleted {deleted_count} old image(s)")
            else:
                print(f"[CLEANUP] No old images to delete")
                
        except Exception as e:
            print(f"[CLEANUP] Error in cleanup task: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup"""
    print(f"[STARTUP] Starting cleanup task (retention: {IMAGE_RETENTION_MINUTES}h, interval: {CLEANUP_INTERVAL_MINUTES}m)")
    asyncio.create_task(cleanup_old_images())

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.api_route("/download/{filename}", methods=["GET", "HEAD"])
async def download_image(filename: str):
    """Serve generated images"""
    file_path = os.path.join("temp_images", filename)
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            media_type='image/jpeg',
            filename=f"ai-photobooth-{filename}"
        )
    else:
        raise HTTPException(status_code=404, detail="Image not found")
async def process_generation(job_id: str, image1_data: bytes, image2_data: bytes):
    """Background task to generate AI image"""
    try:
        print(f"Starting generation for job {job_id}")
        generation_jobs[job_id] = {"status": "processing"}
        
        # Generate the AI image using Gemini
        print("About to call AI generator...")
        generator = get_ai_generator()
        generated_image = await generator.generate_photobooth_image(image1_data, image2_data)
        print(f"AI generator returned: {type(generated_image)}")
        print(f"Generated image size: {len(generated_image) if generated_image else 0} bytes")
        
        print(f"AI generation complete for job {job_id}")
        
        # Upload to local storage
        blob_name = storage_service.upload_generated_image(generated_image, job_id)
        print(f"Saved locally: {blob_name}")
        
        # Get local URL
        download_url = storage_service.get_signed_url(blob_name)
        print(f"Generated local URL for job {job_id}")
        
        # Update job status
        generation_jobs[job_id] = {
            "status": "completed",
            "download_url": download_url
        }
        
    except Exception as e:
        print(f"Generation failed for job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        generation_jobs[job_id] = {"status": "failed", "error": str(e)}
        
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a single file and return public URL"""
    try:
        print(f"Uploading file: {file.filename}")
        
        # Save file with unique name
        filename = f"upload_{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join("temp_images", filename)
        
        file_data = await file.read()
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        # Return full public URL (adjust YOUR_DOMAIN when deployed)
        # Get the base URL dynamically
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        file_url = f"{base_url}/temp/{filename}"
        print(f"File uploaded with URL: {file_url}")
        
        return {"file_url": file_url}
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/temp/{filename}")
async def serve_temp_file(filename: str):
    """Serve temporary uploaded files"""
    file_path = os.path.join("temp_images", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/generate-photobooth")
async def generate_photobooth_image(
    background_tasks: BackgroundTasks,
    request: dict = Body(...)
):
    """Generate photobooth image from two URLs"""
    try:
        image_url_1 = request.get('image_url_1')
        image_url_2 = request.get('image_url_2')
        
        if not image_url_1 or not image_url_2:
            raise HTTPException(status_code=400, detail="Both image_url_1 and image_url_2 are required")
        
        print(f"Generating photobooth from URLs: {image_url_1}, {image_url_2}")
        
        # Extract filenames from URLs and read files
        filename1 = image_url_1.split('/')[-1]
        filename2 = image_url_2.split('/')[-1]
        
        image1_path = os.path.join("temp_images", filename1)
        image2_path = os.path.join("temp_images", filename2)
        
        with open(image1_path, 'rb') as f:
            image1_data = f.read()
        with open(image2_path, 'rb') as f:
            image2_data = f.read()
        
        # Generate job ID and process
        job_id = str(uuid.uuid4())
        
        # Process synchronously for simpler API
        generator = get_ai_generator()  # This will raise an error if it fails
        generated_image = await generator.generate_photobooth_image(image1_data, image2_data)
        
        # Save generated image
        generated_filename = f"generated_{job_id}.jpg"
        generated_path = os.path.join("temp_images", generated_filename)
        with open(generated_path, 'wb') as f:
            f.write(generated_image)
        
        # Return full public URL
        # Get the base URL dynamically
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        generated_url = f"{base_url}/download/{generated_filename}"
        print(f"Generated image available at: {generated_url}")
        
        return {"generated_image_url": generated_url}
        
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image_from_urls(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    file_urls: str = Form(...)
):
    """Generate image from file URLs - matches React GenerateImage function"""
    try:
        # Parse the file_urls JSON string
        file_url_list = json.loads(file_urls)
        print(f"Generating image with prompt: {prompt}")
        print(f"File URLs: {file_url_list}")
        
        # Read files from temp storage
        image1_filename = file_url_list[0].replace("/temp/", "")
        image2_filename = file_url_list[1].replace("/temp/", "")
        
        image1_path = os.path.join("temp_images", image1_filename)
        image2_path = os.path.join("temp_images", image2_filename)
        
        with open(image1_path, 'rb') as f:
            image1_data = f.read()
        with open(image2_path, 'rb') as f:
            image2_data = f.read()
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Start background generation
        background_tasks.add_task(process_generation, job_id, image1_data, image2_data)
        
        # Return the future download URL
        download_url = f"/download/generated_{job_id}.jpg"
        print(f"Will generate image at: {download_url}")
        
        return {"url": download_url}
        
    except Exception as e:
        print(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/polished", response_class=HTMLResponse)
async def polished_design(request: Request):
    """Serve the polished design"""
    return templates.TemplateResponse("polished.html", {"request": request})

@app.post("/upload")
async def upload_and_generate(
    background_tasks: BackgroundTasks,
    image1: UploadFile = File(...), 
    image2: UploadFile = File(...)
):
    """Handle dual image upload and start AI generation"""
    
    # Validate file types
    if not image1.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File 1 must be an image")
    if not image2.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File 2 must be an image")
    
    try:
        print(f"Received upload: {image1.filename}, {image2.filename}")
        
        # Read image data
        image1_data = await image1.read()
        image2_data = await image2.read()
        
        print(f"File sizes: {len(image1_data)}, {len(image2_data)} bytes")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        print(f"Created job: {job_id}")
        
        # Start background generation
        background_tasks.add_task(process_generation, job_id, image1_data, image2_data)
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/polished", response_class=HTMLResponse)
async def polished_design(request: Request):
    """Serve the polished design"""
    return templates.TemplateResponse("polished.html", {"request": request})


@app.get("/status/{job_id}")
async def check_status(job_id: str):
    """Check generation status"""
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = generation_jobs[job_id]
    print(f"Status check for {job_id}: {status}")
    return status

@app.get("/cleanup/stats")
async def cleanup_stats():
    """Get statistics about stored images"""
    try:
        if not os.path.exists(TEMP_IMAGES_DIR):
            return {"total_files": 0, "total_size_mb": 0, "oldest_file_age_hours": 0}
        
        files = [f for f in os.listdir(TEMP_IMAGES_DIR) if os.path.isfile(os.path.join(TEMP_IMAGES_DIR, f))]
        total_size = sum(os.path.getsize(os.path.join(TEMP_IMAGES_DIR, f)) for f in files)
        
        current_time = time.time()
        oldest_age = 0
        if files:
            oldest_file = min(files, key=lambda f: os.path.getmtime(os.path.join(TEMP_IMAGES_DIR, f)))
            oldest_mtime = os.path.getmtime(os.path.join(TEMP_IMAGES_DIR, oldest_file))
            oldest_age = (current_time - oldest_mtime) / 3600  # Convert to hours
        
        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_file_age_hours": round(oldest_age, 2),
            "retention_hours": IMAGE_RETENTION_MINUTES,
            "cleanup_interval_minutes": CLEANUP_INTERVAL_MINUTES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup/manual")
async def manual_cleanup():
    """Manually trigger cleanup of old images"""
    try:
        current_time = time.time()
        retention_seconds = IMAGE_RETENTION_MINUTES * 60
        deleted_count = 0
        deleted_files = []
        
        if not os.path.exists(TEMP_IMAGES_DIR):
            return {"deleted_count": 0, "message": "Temp directory does not exist"}
        
        for filename in os.listdir(TEMP_IMAGES_DIR):
            file_path = os.path.join(TEMP_IMAGES_DIR, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            file_mtime = os.path.getmtime(file_path)
            file_age_seconds = current_time - file_mtime
            
            if file_age_seconds > retention_seconds:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    deleted_files.append({
                        "filename": filename,
                        "age_hours": round(file_age_seconds / 3600, 2)
                    })
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
        
        return {
            "deleted_count": deleted_count,
            "deleted_files": deleted_files,
            "message": f"Deleted {deleted_count} file(s) older than {IMAGE_RETENTION_MINUTES} hours"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
