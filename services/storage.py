import os
import uuid
from typing import Optional

class CloudStorageService:
    def __init__(self):
        # Create a local temp directory for storing images
        self.temp_dir = "temp_images"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        print(f"✅ Using local storage in: {self.temp_dir}")
    
    def upload_image(self, image_data: bytes, filename: str) -> str:
        """Save image locally and return the filename"""
        blob_name = f"uploads_{uuid.uuid4()}_{filename}"
        file_path = os.path.join(self.temp_dir, blob_name)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        return blob_name
    
    def upload_generated_image(self, image_data: bytes, job_id: str) -> str:
        """Save generated image locally"""
        blob_name = f"generated_{job_id}.jpg"
        file_path = os.path.join(self.temp_dir, blob_name)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        print(f"Saved image locally: {file_path}")
        return blob_name
    
    def get_signed_url(self, blob_name: str, expiration_hours: int = 24) -> str:
        """Generate a local URL for downloading"""
        return f"/download/{blob_name}"
