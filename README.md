# transcribe-worker
Worker for the SUNET transcription service

## Development environment setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
	```bash
	source venv/bin/activate
	```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Edit the environment settings, should be in a file named `.env`. The following settings should be sufficient for most cases:
	```env
	DEBUG=True
	API_BACKEND_URL="http://localhost:8000"
	API_VERSION="v1"
	WORKERS=2
	WHISPER_CPP_PATH=<Path to whisper.cpp>
	FILE_STORAGE_DIR="<Your file storage directory>"
	```

5. Build and install whisper.cpp, see https://github.com/ggml-org/whisper.cpp for details.

6. Download the needed Whisper models. From the transcriber-worker directory run:
	```bash
	./download_models.sh
	```

7. Run the worker:
	```bash
	python3 main.py --foreground --debug
	```

