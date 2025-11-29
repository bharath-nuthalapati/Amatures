# Application Architecture & Workflow

## 1. 🎯 The Purpose
The **Universal Accessibility Reader** is designed to make reading accessible to everyone, specifically targeting:
-   **Dyslexia**: By simplifying complex sentences and providing specialized fonts.
-   **Low Vision**: By extracting text from images/PDFs and reading it aloud.
-   **Learning Differences**: By converting academic or dense text into simple, easy-to-understand language.

## 2. 🔄 The Workflow (How it Works)
The application follows a linear 5-step pipeline:

1.  **Input**
    -   User uploads an **Image** (PNG/JPG), a **PDF**, or pastes raw **Text**.
    -   *Goal*: Get the raw content into the system.

2.  **OCR (Optical Character Recognition)**
    -   The app processes the image to extract text.
    -   *Tech*: **Google Gemini 2.0 Flash**. This is a multimodal model that can "see" images and documents with high accuracy.

3.  **Simplification**
    -   The extracted text is sent to a local AI model to be rewritten based on the selected level:
        -   *Light*: Fixes grammar and clarity.
        -   *Medium*: Uses simpler words and shorter sentences.
        -   *Strong*: Explains concepts as if to a child (Grade 3-4 level).
    -   *Tech*: **FLAN-T5-Small**. This runs locally on your CPU, ensuring privacy and speed without external API costs for this step.

4.  **Text-to-Speech (TTS)**
    -   The simplified text is converted into natural-sounding audio.
    -   *Tech*: 
        -   **ElevenLabs API** (Primary): Provides high-quality, human-like voices with emotion (Happy, Calm, Neutral).
        -   **gTTS** (Fallback): Uses Google Translate's free TTS if no ElevenLabs API key is provided.

5.  **Output & History**
    -   The UI displays the Original Text vs. Simplified Text side-by-side.
    -   An audio player allows listening or downloading the MP3.
    -   The session is saved to a local JSON history file for later retrieval.

## 3. 🛠️ The Tech Stack
This is a production-ready Python application built with:

*   **Frontend UI**: `Gradio`
    -   Builds the accessible web interface with tabs, file uploads, and interactive components.
*   **Vision/OCR**: `Google Gemini 2.0 Flash` (via `google-generativeai`)
    -   Handles the complex task of reading text from images.
*   **Text Processing**: `HuggingFace Transformers` (`google/flan-t5-small`)
    -   Performs the intelligent text simplification locally.
*   **Audio Generation**: `ElevenLabs API` & `gTTS`
    -   Powers the voice synthesis.
*   **Backend Logic**: `Python 3.14`
    -   Orchestrates the entire pipeline.
*   **Storage**: `JSON`
    -   A lightweight, file-based database to store user history without needing a complex server setup.

## 🚀 Quick Summary
**Upload File** → **Gemini Reads It** → **T5 Simplifies It** → **ElevenLabs Speaks It** → **You Read/Listen**
