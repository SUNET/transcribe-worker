# transcribe-worker
Worker for the SUNET transcription service

## Development environment setup

1. Edit the environment settings, should be in a file named `.env`. The following settings should be sufficient for most cases:
	```env
	DEBUG=True
	API_BACKEND_URL="http://localhost:8000"
	API_VERSION="v1"
	WORKERS=2
	WHISPER_CPP_PATH=<Path to whisper.cpp>
	FILE_STORAGE_DIR="<Your file storage directory>"
	```

2. Build and install whisper.cpp, see https://github.com/ggml-org/whisper.cpp for details.

3. Download the needed Whisper models. From the transcriber-worker directory run:
	```bash
	./download_models.sh
	```

4. Run the worker with uv:
	```bash
	uv run main.py --foreground --debug
	```

