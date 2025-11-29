
import gradio as gr
import os
import modal
from mcp_server import MCPServer
from utils import text_to_speech, map_sound_to_vibration
import time

# Initialize MCP Server
mcp = MCPServer()

# Modal Function Lookup
# Assumes 'modal deploy modal_app.py' has been run
try:
    ModelInference = modal.Cls.lookup("accessibility-companion", "ModelInference")
    model_inference = ModelInference()
except Exception as e:
    print(f"Could not lookup Modal app: {e}")
    print("Ensure you have deployed the app with `modal deploy modal_app.py`")
    model_inference = None

def process_vision(image):
    if not image:
        return "No image provided.", "No image provided."
    
    if not model_inference:
        return "Backend not connected.", "Backend not connected."
    
    # Convert to bytes for Modal
    import io
    from PIL import Image
    img_byte_arr = io.BytesIO()
    Image.fromarray(image).save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    # Run inference
    ocr_text = model_inference.process_image_ocr.remote(img_bytes)
    scene_data = model_inference.detect_objects.remote(img_bytes)
    
    # Format Scene Description
    scene_desc = "Objects detected:\n"
    for obj in scene_data.get("objects", []):
        scene_desc += f"- {obj['label']} ({int(obj['score']*100)}%)\n"
        
    # Simplify text if needed (using the same model for now)
    simplified_text = model_inference.simplify_text.remote(ocr_text)
    
    # TTS
    audio = text_to_speech(simplified_text)
    if audio:
        # Save audio to file for Gradio
        with open("output_tts.mp3", "wb") as f:
            f.write(audio)
        return simplified_text, scene_desc, "output_tts.mp3"
    
    return simplified_text, scene_desc, None

def process_audio(audio_path):
    if not audio_path:
        return "No audio input.", ""
    
    # Local Whisper (using openai-whisper package if installed, or API)
    # For this demo, we'll simulate or use a small model if available.
    # To keep it simple and fast, let's assume we use the Modal backend or a placeholder
    # But user asked for "Whisper model (tiny or base) locally"
    
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path)
        text = result["text"]
        
        # Sound Classification Simulation (mapping text keywords to sounds for demo)
        # In a real app, we'd run a classifier on the audio waveform.
        haptic_feedback = "No specific sound pattern detected."
        text_lower = text.lower()
        
        detected_sound = None
        if "bark" in text_lower: detected_sound = "dog_bark"
        elif "horn" in text_lower: detected_sound = "car_horn"
        elif "alarm" in text_lower: detected_sound = "alarm"
        elif "door" in text_lower: detected_sound = "doorbell"
        
        if detected_sound:
            pattern = map_sound_to_vibration(detected_sound)
            haptic_feedback = f"Vibration: {pattern['pattern'].upper()} (Intensity: {pattern['intensity']})"
            
        return text, haptic_feedback
        
    except ImportError:
        return "Whisper not installed.", "Install openai-whisper to enable STT."
    except Exception as e:
        return f"Error: {str(e)}", ""

# MCP Wrappers
def run_mcp_tool(tool_name):
    if tool_name == "Calendar":
        return mcp.get_calendar_events()
    elif tool_name == "Email":
        return mcp.summarize_emails()
    elif tool_name == "Maps":
        return mcp.navigate_maps("Home")
    return "Unknown tool"

# Custom CSS for Accessibility
custom_css = """
.gradio-container { font-family: 'Arial', sans-serif; }
button.primary { background-color: #0056b3 !important; font-size: 1.2em !important; }
label { font-size: 1.1em !important; font-weight: bold !important; }
"""

def main():
    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Access Companion") as demo:
        gr.Markdown("# ♿ Universal Accessibility Companion")
        
        with gr.Tabs():
            # --- Vision Tab ---
            with gr.Tab("👁️ Vision Assistant"):
                gr.Markdown("### Capture scene or document")
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(type="numpy", label="Camera View", sources=["webcam", "upload"])
                        process_btn = gr.Button("Describe Scene & Read Text", variant="primary")
                    
                    with gr.Column():
                        ocr_output = gr.Textbox(label="Simplified Text (Voice Output)", lines=4)
                        scene_output = gr.Textbox(label="Scene Objects", lines=4)
                        audio_output = gr.Audio(label="TTS Output", type="filepath")
                
                process_btn.click(
                    process_vision, 
                    inputs=[img_input], 
                    outputs=[ocr_output, scene_output, audio_output]
                )
            
            # --- Hearing Tab ---
            with gr.Tab("👂 Hearing Assistant"):
                gr.Markdown("### Live Captioning & Sound Sensing")
                with gr.Row():
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Microphone Input")
                    with gr.Column():
                        captions = gr.Textbox(label="Live Captions", lines=2)
                        haptics_indicator = gr.Textbox(label="Haptic Feedback Simulator", lines=1)
                
                audio_input.change(
                    process_audio,
                    inputs=[audio_input],
                    outputs=[captions, haptics_indicator]
                )
            
            # --- Integrations Tab ---
            with gr.Tab("🔗 Integrations (MCP)"):
                gr.Markdown("### Connected Services")
                with gr.Row():
                    cal_btn = gr.Button("📅 Check Calendar")
                    email_btn = gr.Button("📧 Read Emails")
                    maps_btn = gr.Button("🗺️ Navigate Home")
                
                tool_output = gr.Textbox(label="System Response", lines=5)
                
                cal_btn.click(lambda: run_mcp_tool("Calendar"), outputs=tool_output)
                email_btn.click(lambda: run_mcp_tool("Email"), outputs=tool_output)
                maps_btn.click(lambda: run_mcp_tool("Maps"), outputs=tool_output)

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
