import os
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from pdf2image import convert_from_path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
HISTORY_FILE = "history.json"
CACHE_DIR = "cache"
AUDIO_DIR = "audio_output"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)



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

            if input_type == "PDF":
                images = convert_from_path(file_path)
                if not images:
                    return "Error: Empty PDF"
                image = images[0]
            else:
                image = Image.open(file_path)

            prompt = "Extract all text from this image exactly as it appears. Do not add any commentary."
            
            response = self.model.generate_content([prompt, image])
            
            return response.text.strip()
                
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return f"Error extracting text: {str(e)}"

class SimplificationEngine:
    def __init__(self):
        self.device = "cpu" # Force CPU for lightweight requirement
        self.model_name = "google/flan-t5-small"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            logger.info("Simplification model loaded.")
        except Exception as e:
            logger.error(f"Failed to load primary model: {e}")
            # Fallback could be implemented here
            
    def simplify(self, text: str, level: str) -> str:
        if not text:
            return ""
            
        prompts = {
            "Light": "Fix grammar and clarity: ",
            "Medium": "Simplify this text for a general audience: ",
            "Strong": "Explain this like I am 8 years old: "
        }
        
        prompt_prefix = prompts.get(level, prompts["Medium"])
        
        # Chunk text to handle model limits
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        simplified_chunks = []
        
        for chunk in chunks:
            input_text = prompt_prefix + chunk
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            
            outputs = self.model.generate(
                input_ids, 
                max_length=512, 
                num_beams=4, 
                early_stopping=True
            )
            
            simplified_chunks.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            
        return " ".join(simplified_chunks)

from gtts import gTTS

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

class TTSEngine:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if self.api_key:
            self.client = ElevenLabs(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("ELEVENLABS_API_KEY not found. Switching to gTTS (Free Fallback).")

    def generate_audio(self, text: str, tone: str) -> Optional[str]:
        if not text:
            return None
            
        filename = f"audio_{int(time.time())}.mp3"
        filepath = os.path.join(AUDIO_DIR, filename)

        # 1. Try ElevenLabs if Key exists
        if self.client:
            try:
                # Map tones to stability/similarity settings
                voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM") # Default to Rachel
                
                settings = VoiceSettings(stability=0.5, similarity_boost=0.75)
                if tone == "Calm":
                    settings = VoiceSettings(stability=0.8, similarity_boost=0.5)
                elif tone == "Happy":
                    settings = VoiceSettings(stability=0.3, similarity_boost=0.8)
                    
                audio_generator = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_monolingual_v1",
                    voice_settings=settings
                )
                
                # Consume generator and write to file
                with open(filepath, "wb") as f:
                    for chunk in audio_generator:
                        f.write(chunk)
                    
                return filepath
            except Exception as e:
                logger.error(f"ElevenLabs generation failed: {e}. Falling back to gTTS.")
        
        # 2. Fallback to gTTS (Free)
        try:
            # Map tone to speed/lang roughly (gTTS is limited)
            # We can't change tone easily in gTTS, but we can ensure it works
            tts = gTTS(text=text, lang='en', slow=(tone == "Calm"))
            tts.save(filepath)
            return filepath
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return None

class HistoryManager:
    def __init__(self):
        self.file_path = HISTORY_FILE
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def add_entry(self, entry: Dict[str, Any]):
        with open(self.file_path, 'r') as f:
            history = json.load(f)
        
        history.insert(0, entry) # Add to top
        
        with open(self.file_path, 'w') as f:
            json.dump(history, f, indent=2)

    def get_history(self) -> List[Dict[str, Any]]:
        with open(self.file_path, 'r') as f:
            return json.load(f)

# --- Main Application Logic ---

ocr_engine = OCREngine()
simplification_engine = SimplificationEngine()
tts_engine = TTSEngine()
history_manager = HistoryManager()

def process_content(file_obj, input_text, input_type, simplify_level, voice_tone):
    # 1. Get Text
    extracted_text = ""
    source_name = "Text Input"
    
    if input_type == "Text":
        extracted_text = input_text
    elif file_obj is not None:
        source_name = os.path.basename(file_obj.name)
        extracted_text = ocr_engine.extract_text(file_obj.name, input_type)
    else:
        return "Please provide input.", "", None

    # 2. Simplify
    simplified_text = simplification_engine.simplify(extracted_text, simplify_level)
    
    # 3. Generate Audio
    audio_path = tts_engine.generate_audio(simplified_text, voice_tone)
    
    # 4. Save History
    session_id = hashlib.md5(f"{source_name}{time.time()}".encode()).hexdigest()[:8]
    history_entry = {
        "id": session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "filename": source_name,
        "original_text": extracted_text,
        "simplified_text": simplified_text,
        "tone": voice_tone,
        "audio_path": audio_path
    }
    history_manager.add_entry(history_entry)
    
    return extracted_text, simplified_text, audio_path

def load_history_view():
    history = history_manager.get_history()
    # Convert to list of lists for Dataframe
    data = []
    for h in history:
        data.append([
            h['id'],
            h['timestamp'],
            h['filename'],
            h['simplified_text'][:50] + "..."
        ])
    return data

def restore_session(evt: gr.SelectData):
    # Get full history to find the selected entry
    history = history_manager.get_history()
    # The selected row index corresponds to the history index (since we display all)
    if evt.index[0] < len(history):
        entry = history[evt.index[0]]
        return (
            entry['original_text'],
            entry['simplified_text'],
            entry['audio_path']
        )
    return "", "", None

# --- Gradio UI ---

custom_css = """
.container { max-width: 1200px; margin: auto; }
.dyslexic-font { font-family: 'Open Dyslexic', sans-serif !important; }
"""

with gr.Blocks(title="Universal Accessibility Reader") as app:
    gr.Markdown("# ⭐ Universal Accessibility Reader")
    
    with gr.Tabs():
        # Tab 1: Read & Listen
        with gr.Tab("📄 Read & Listen"):
            with gr.Row():
                # Left Panel: Controls
                with gr.Column(scale=1):
                    gr.Markdown("### Input Settings")
                    
                    input_type = gr.Dropdown(
                        choices=["Image", "PDF", "Text"], 
                        value="Image", 
                        label="Input Type"
                    )
                    
                    file_input = gr.File(label="Upload File", file_types=[".png", ".jpg", ".pdf"])
                    text_input = gr.Textbox(label="Or Paste Text", lines=5, visible=False)
                    
                    def update_input_visibility(choice):
                        return {
                            file_input: gr.update(visible=choice != "Text"),
                            text_input: gr.update(visible=choice == "Text")
                        }
                    
                    input_type.change(update_input_visibility, input_type, [file_input, text_input])
                    
                    simplify_level = gr.Slider(
                        minimum=0, maximum=2, step=1, 
                        label="Simplification Strength",
                        info="0: Light, 1: Medium, 2: Strong"
                    )
                    # Map slider to string labels for backend
                    level_map = {0: "Light", 1: "Medium", 2: "Strong"}
                    
                    voice_tone = gr.Dropdown(
                        choices=["Calm", "Happy", "Neutral"], 
                        value="Neutral", 
                        label="Voice Tone"
                    )
                    
                    process_btn = gr.Button("Process & Generate Audio", variant="primary")
                    
                    gr.Markdown("### Accessibility Options")
                    dyslexia_mode = gr.Checkbox(label="Dyslexia-Friendly Mode")
                    
                # Right Panel: Output
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### Extracted Text")
                        output_original = gr.Textbox(
                            label="Original", 
                            lines=6, 
                            interactive=False
                        )
                    
                    with gr.Group():
                        gr.Markdown("### Simplified Version")
                        output_simplified = gr.Textbox(
                            label="Simplified", 
                            lines=6, 
                            interactive=False
                        )
                    
                    audio_player = gr.Audio(label="Listen", type="filepath")

            # Processing Logic
            def on_process(file, txt, type_in, level_idx, tone):
                level = level_map[level_idx]
                return process_content(file, txt, type_in, level, tone)

            process_btn.click(
                on_process,
                inputs=[file_input, text_input, input_type, simplify_level, voice_tone],
                outputs=[output_original, output_simplified, audio_player]
            )

        # Tab 2: History
        with gr.Tab("📁 History"):
            gr.Markdown("### Recent Sessions")
            refresh_btn = gr.Button("Refresh History")
            
            history_table = gr.Dataframe(
                headers=["ID", "Timestamp", "File", "Preview"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )
            
            refresh_btn.click(load_history_view, outputs=history_table)
            
            # Click on table row to restore
            history_table.select(
                restore_session,
                outputs=[output_original, output_simplified, audio_player]
            )
            
            # Auto-load history on launch
            app.load(load_history_view, outputs=history_table)

        # Tab 3: Help
        with gr.Tab("❓ Help & About"):
            gr.Markdown("""
            ### How to Use
            1. **Upload**: Select an image or PDF file.
            2. **Customize**: Choose simplification level and voice tone.
            3. **Process**: Click the button to generate text and audio.
            4. **Listen**: Play the generated audio or download it.
            
            ### Accessibility Features
            - **Dyslexia Mode**: Improves readability with specialized fonts and spacing.
            - **Simplification**: Makes complex text easier to understand.
            """)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
