import re
import os
try:
    from elevenlabs import generate, set_api_key
except ImportError:
    print("ElevenLabs not installed or import failed.")

def text_to_speech(text: str):
    """
    Converts text to speech using ElevenLabs API.
    Returns audio bytes or None if API key is missing.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Warning: ELEVENLABS_API_KEY not set.")
        return None
    
    try:
        set_api_key(api_key)
        audio = generate(
            text=text,
            voice="Bella",
            model="eleven_monolingual_v1"
        )
        return audio
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None


def map_sound_to_vibration(sound_label: str):
    """
    Maps a sound classification label to a vibration pattern.
    Returns a description of the pattern (e.g., 'short-short', 'long').
    """
    mapping = {
        "dog_bark": {"pattern": "short-short", "intensity": "medium"},
        "car_horn": {"pattern": "strong-pulse", "intensity": "high"},
        "alarm": {"pattern": "long-continuous", "intensity": "high"},
        "doorbell": {"pattern": "double-pulse", "intensity": "medium"},
        "siren": {"pattern": "rapid-pulse", "intensity": "high"},
        "cat_meow": {"pattern": "soft-short", "intensity": "low"},
        "glass_breaking": {"pattern": "sharp-pulse", "intensity": "high"}
    }
    return mapping.get(sound_label, {"pattern": "default", "intensity": "low"})

def clean_text(text: str):
    """
    Basic text cleaning.
    """
    return re.sub(r'\s+', ' ', text).strip()
