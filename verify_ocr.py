import os
import logging
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env vars
load_dotenv()

class OCREngine:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not found. OCR will fail.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-flash-latest')
            logger.info("OCR Engine initialized with Google Gemini (gemini-flash-latest)")

    def extract_text(self, file_path: str, input_type: str) -> str:
        """
        Extract text from image or PDF using Google Gemini.
        """
        try:
            if not self.api_key:
                return "Error: GOOGLE_API_KEY not set."

            # Note: PDF support requires pdf2image which we skip for this simple image test
            if input_type == "PDF":
                return "PDF testing skipped in verification script"
            else:
                image = Image.open(file_path)

            prompt = "Extract all text from this image exactly as it appears. Do not add any commentary."
            
            response = self.model.generate_content([prompt, image])
            
            # Debug: Print response details if candidates are empty
            if not response.candidates:
                print(f"DEBUG: Response Feedback: {response.prompt_feedback}")
                print(f"DEBUG: Full Response: {response}")
            
            return response.text.strip()
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return f"Error extracting text: {str(e)}"

def list_models(api_key):
    genai.configure(api_key=api_key)
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

def test_ocr():
    print("Initializing OCREngine...")
    ocr = OCREngine()
    
    # Debug: List models
    # list_models(ocr.api_key)
    
    test_image_path = "venv/lib/python3.14/site-packages/gradio/media_assets/images/cheetah.jpg"
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return

    print(f"Testing OCR with {test_image_path}...")
    try:
        text = ocr.extract_text(test_image_path, "Image")
        print("\n--- OCR Result ---")
        print(text)
        print("------------------\n")
        
        if "Error" in text:
            print("FAILED: OCR returned an error.")
        else:
            print("SUCCESS: OCR extracted text.")
            
    except Exception as e:
        print(f"FAILED: Exception occurred: {e}")

if __name__ == "__main__":
    test_ocr()
