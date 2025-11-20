import modal
import io
from PIL import Image

app = modal.App("accessibility-companion")

# Define image with dependencies
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "pillow",
    "scipy",
    "accelerate",
    "sentencepiece",
    "protobuf"
)

@app.cls(image=image, gpu="any")
class ModelInference:
    def enter(self):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers import OwlViTProcessor, OwlViTForObjectDetection
        from transformers import BartTokenizer, BartForConditionalGeneration
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device}...")

        # OCR Model
        self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(self.device)

        # Object Detection Model
        self.od_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.od_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

        # Text Simplification Model
        self.sim_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.sim_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(self.device)
        
        print("Models loaded.")

    @modal.method()
    def process_image_ocr(self, image_bytes: bytes) -> str:
        from PIL import Image
        import torch
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = self.ocr_processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        generated_ids = self.ocr_model.generate(pixel_values)
        generated_text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    @modal.method()
    def detect_objects(self, image_bytes: bytes) -> dict:
        from PIL import Image
        import torch
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.od_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.od_model(**inputs)
            
        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.od_processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
        
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detected_objects.append({
                "label": self.od_model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": [round(i, 2) for i in box.tolist()]
            })
            
        return {"objects": detected_objects}

    @modal.method()
    def simplify_text(self, text: str) -> str:
        inputs = self.sim_tokenizer([text], max_length=1024, return_tensors="pt").to(self.device)
        summary_ids = self.sim_model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=100)
        return self.sim_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
