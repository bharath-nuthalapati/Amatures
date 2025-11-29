# Universal Accessibility Reader ⭐

A production-ready application designed to assist users with dyslexia, learning differences, visual strain, or low reading confidence. The system processes uploaded content through a pipeline of OCR, text cleaning, simplification, and voice generation.

## Features

- **High-Accuracy OCR**: Uses Gemini 2.5 Flash to extract text from images (PNG, JPG) and PDFs.
- **Text Simplification**: Uses FLAN-T5 to simplify text with three severity levels (Light, Medium, Strong).
- **Natural Voice Generation**: Uses ElevenLabs API for high-quality, natural-sounding speech with multiple tones.
- **Accessibility-First UI**: High-contrast modes, dyslexia-friendly fonts, and adjustable text settings.
- **History & Session Management**: Save and replay previous reading sessions.

## Prerequisites

- Python 3.9+
- `poppler-utils` (for PDF processing)
  - macOS: ``brew install poppler``
  - Ubuntu: `sudo apt-get install poppler-utils`
  - Windows: Download and add to PATH

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd LexiAssist
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys:
     ```
     GOOGLE_API_KEY=your_gemini_api_key
     ELEVENLABS_API_KEY=your_elevenlabs_api_key
     ELEVENLABS_VOICE_ID=your_voice_id
     ```

## Usage

Run the application using the virtual environment:
```bash
./venv/bin/python app.py
```
Open your browser to the local Gradio URL (usually http://127.0.0.1:7860).

## Architecture

The application follows a linear processing pipeline:
1. **Upload**: User uploads an image or PDF.
2. **Preprocessing**: Image is cleaned (deskewed, contrast enhanced).
3. **OCR**: Gemini 2.5 Flash extracts text.
4. **Simplification**: FLAN-T5 simplifies text based on user preference.
5. **TTS**: ElevenLabs generates audio.
6. **Storage**: Session is saved to local history.

## Accessibility Options

- **Dyslexia-Friendly Mode**: Adjusts font, spacing, and line width.
- **High Contrast**: Improves visibility for low-vision users.
- **Voice Tones**: Calm, Happy, and Neutral options for different contexts.
