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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
