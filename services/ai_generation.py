import os
from google import genai
import base64
import uuid
from PIL import Image
import io

class PhotoBoothGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    
    def convert_to_jpeg(self, image_data: bytes) -> bytes:
        """Convert any image format to JPEG"""
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85)
            return output.getvalue()
        except Exception as e:
            print(f"Image conversion error: {e}")
            return image_data
    
    async def generate_photobooth_image(self, image1_data: bytes, image2_data: bytes) -> bytes:
        """Generate a photobooth style image from two selfies"""
        
        print(f"Starting AI generation with images of size: {len(image1_data)}, {len(image2_data)} bytes")
        
        # Convert both images to JPEG first
        image1_jpeg = self.convert_to_jpeg(image1_data)
        image2_jpeg = self.convert_to_jpeg(image2_data)
        print(f"Converted to JPEG, sizes: {len(image1_jpeg)}, {len(image2_jpeg)} bytes")
        
        # Convert images to base64 for Gemini
        image1_b64 = base64.b64encode(image1_jpeg).decode()
        image2_b64 = base64.b64encode(image2_jpeg).decode()
        
        prompt = """Create a vintage black-and-white photobooth strip, composed of exactly four distinct frames each featureing person A and person B stacked to a 2x2 grid square format, Each frame should be high-contrast black-and-white with authentic film grain and soft photobooth lighting.

For the first pair of subjects (from the provided selfies):

Frame 1 (Top): Both making a kissy face.

Frame 2: Both smiling wide, one flashing a peace sign.

Frame 3: Both looking serious and shocked.

Frame 4 (Bottom): Both leaning close, laughing naturally.

Ensure a thin white border separates each of the four frames. The poses should be playful and capture the natural interaction of two close friends."""
        
        try:
            print("Calling Gemini API...")
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-image-preview',
                contents=[
                    prompt,
                    {"inline_data": {"mime_type": "image/jpeg", "data": image1_b64}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image2_b64}}
                ]
            )
            
            # Try to get the image directly from the response object
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        print(f"Found image part with data length: {len(part.inline_data.data)}")
                        
                        # Try direct access to the data attribute
                        image_data = part.inline_data.data
                        
                        # Check if it's already bytes or needs base64 decoding
                        if isinstance(image_data, bytes):
                            print(f"Image data is already bytes: {len(image_data)}")
                            return image_data
                        else:
                            print(f"Decoding base64 data of length: {len(image_data)}")
                            decoded = base64.b64decode(image_data)
                            print(f"Decoded to {len(decoded)} bytes")
                            return decoded
            
            raise Exception("No image found in response")
            
        except Exception as e:
            print(f"Error in AI generation: {e}")
            raise
