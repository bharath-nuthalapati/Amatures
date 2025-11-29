---
description: Setup and run the Universal Accessibility Reader
---

1. Install system dependencies (macOS)
// turbo
brew install poppler

2. Create virtual environment
// turbo
python3 -m venv venv

3. Install Python dependencies
// turbo
./venv/bin/pip install -r requirements.txt

4. Configure environment variables
cp .env.example .env
# Note: You must manually edit .env to add your API keys

5. Run the application
./run.sh
