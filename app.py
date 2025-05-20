import logging
import requests
import subprocess
import traceback
import threading

from enum import Enum
from settings import get_settings
from time import sleep
from pathlib import Path
from random import randint


class JobStatusEnum(str, Enum):
    """
    Enum representing the status of a job.
    """

    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


settings = get_settings()

api_broker_url = settings.API_BROKER_URL
api_file_storage_dir = settings.API_FILE_STORAGE_DIR
api_version = settings.API_VERSION
api_url = f"{api_broker_url}/api/{api_version}/transcriber"


def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


logger = get_logger()


def transcode_file(filename: str):
    """
    Transcode the audio file using ffmpeg.
    The transcoded format should be 16kHz mono WAV.
    """

    output_filename = f"{filename}.wav"

    command = [
        "ffmpeg",
        "-i",
        str(Path(api_file_storage_dir) / filename),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        "-y",
        str(Path(api_file_storage_dir) / output_filename),
    ]

    try:
        ffmpeg_cmd = " ".join(command)
        logger.debug(f"Transcoding command: {ffmpeg_cmd}")
        result = subprocess.run(command, check=True, capture_output=True)

        # Check exit code
        if result.returncode != 0:
            logger.error(f"Error during transcoding: {result.stderr.decode()}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during transcoding: {e}")
        raise e
    else:
        logger.info(f"Transcoding completed: {output_filename}")

    return True


def transcribe_file(filename: str, language: str, model: str, output_format: str):
    """
    Transcribe the audio file using whisper.cpp, we expect the executable
    to be in PATH.
    """
    command = [
        "whisper.cpp",
        "-l",
        language,
        f"-o{output_format.lower()}",
        "-of",
        str(Path(api_file_storage_dir) / filename),
        "-m",
        model,
        "-f",
        str(Path(api_file_storage_dir) / f"{filename}.wav"),
    ]

    try:
        whisper_cmd = " ".join(command)
        logger.debug(f"Transcription command: {whisper_cmd}")
        result = subprocess.run(command, check=True, capture_output=True)

        # Check exit code
        if result.returncode != 0:
            logger.error(f"Error during transcription: {result.stderr.decode()}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during transcription: {e}")
        raise e
    else:
        logger.info(f"Transcription completed: {filename}")

    return True


def get_next_job(url: str) -> dict:
    """
    Get the next job from the API broker.
    """
    response = requests.get(f"{api_url}/next")
    response.raise_for_status()

    job = response.json()["result"]

    if job == {}:
        return {}
    if job["status"] != JobStatusEnum.IN_PROGRESS:
        logger.info(f"Job {job['uuid']} is not in_progress. Skipping.")
        return {}

    return job


def get_file(uuid: str) -> bool:
    """
    Download the file from the API broker.
    """

    response = requests.get(f"{api_url}/{uuid}/file")
    response.raise_for_status()

    if response.status_code != 200:
        logger.error(f"Error downloading file: {response.status_code}")
        raise Exception("File not downloaded")

    file_path = Path(api_file_storage_dir) / uuid

    with open(file_path, "wb") as f:
        f.write(response.content)

    return True


def put_status(uuid: str, status: JobStatusEnum, error: str) -> bool:
    """
    Update the job status in the API broker.
    """
    try:
        response = requests.put(
            f"{api_url}/{uuid}", json={"status": status, "error": error}
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error updating job status: {e}")
        return False
    return True


def put_file(uuid: str, output_format: str) -> bool:
    """
    Upload the file to the API broker.
    """

    file_path = Path(api_file_storage_dir) / f"{uuid}.{output_format}"
    with open(file_path, "rb") as fd:
        response = requests.put(f"{api_url}/{uuid}/result", files={"file": fd})
        response.raise_for_status()


def get_model(model_type: str, language: str) -> str:
    """
    Return the correct model file based on
    model type and language.

    If langauge = sv then use kb-whisper else
    use the default whisper model.
    """

    file_path = "models/"
    file_path += "sv" if language == "sv" else "en"

    match model_type:
        case "tiny":
            file_path += "_tiny"
        case "base":
            file_path += "_base"
        case "large":
            file_path += "_large"
        case _:
            file_path += "_base"

    return f"{file_path}.bin"


def postprocess_srt(uuid: str, output_format: str) -> bool:
    """
    Postprocess the SRT file to remove unwanted characters.
    """

    # Timing based on reading speed
    # • Default reading speed: 160–180 words per minute.
    # • Minimum display time: 1.5 seconds, to avoid flickering subtitles.

    # Audio-sync and natural timing
    # • Subtitle should ideally appear 0.5 seconds before speech and remain at least 1 second after the final word.
    # • If possible, adjust timestamps for natural rhythm in relation to spoken tempo.

    # Export in professional formats
    # • Primary export to .srt.
    # • Support for .vtt or possibly .json if needed for further processing.

    if output_format != "srt":
        return False

    srt_path = Path(api_file_storage_dir) / f"{uuid}.srt"
    with open(srt_path, "r") as f:
        content = f.read()

    for line in content.splitlines():
        if len(line) > 42:
            # Split line into two lines if longer than 42 characters
            split_line = line[:42] + "\n" + line[42:]
            content = content.replace(line, split_line)

    # • Avoid line breaks in the middle of names or fixed expressions.
    # Remove unwanted characters

    with open(srt_path, "w") as f:
        f.write(content)

    logger.info(f"Postprocessing completed for {uuid}.srt")
    return True


def delete_files(uuid: str) -> bool:
    """
    Delete all files related to the job.
    """
    file_path = Path(api_file_storage_dir) / uuid
    if file_path.exists():
        file_path.unlink()
        logger.info(f"Deleted file {file_path}")

    wav_file_path = Path(api_file_storage_dir) / f"{uuid}.wav"
    if wav_file_path.exists():
        wav_file_path.unlink()
        logger.info(f"Deleted file {wav_file_path}")

    srt_file_path = Path(api_file_storage_dir) / f"{uuid}.srt"
    if srt_file_path.exists():
        srt_file_path.unlink()
        logger.info(f"Deleted file {srt_file_path}")

    return True


def main(worker_id: int):
    """
    Main function to fetch jobs and process them.
    """
    logger.info(
        f"[{worker_id}] Starting transcription service, server URL: {api_broker_url}"
    )

    while True:
        try:
            # Sleep for a random time between 5 and 10 seconds
            # to avoid hammering the API broker.
            sleep_time = randint(5, 10)
            logger.debug(
                f"[{worker_id}] Sleeping for {sleep_time} seconds before checking for jobs."
            )
            sleep(sleep_time)

            if not (job := get_next_job(api_broker_url)):
                continue

            uuid = job["uuid"]
            language = job["language"]
            model_type = job["model_type"]
            model = get_model(model_type, language)
            output_format = job["output_format"]

            logger.info(f"[{worker_id}] Processing job {uuid}:")
            logger.info(f"  Language: {language}")
            logger.info(f"  Model Type: {model_type}")
            logger.info(f"  Model: {model}")
            logger.info(f"  Output Format: {output_format}")

            # Download the file
            get_file(uuid)

            # Transcode the file
            transcode_file(uuid)

            # Transcribe the file
            transcribe_file(uuid, language, model, output_format)

            # Postprocess subtitles
            # postprocess_srt(uuid, output_format)

            # Upload the resulting SRT
            put_file(f"{uuid}", output_format)

            # Remove all files
            delete_files(uuid)

            logger.info(f"[{worker_id}] Job {uuid} completed successfully.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[{worker_id}] Connection error: {e}")
            continue
        except requests.exceptions.HTTPError as e:
            logger.error(f"[{worker_id}] HTTP error: {e}")
            continue
        except Exception as e:
            put_status(uuid, JobStatusEnum.FAILED, error=str(e))
            logger.error(f"[{worker_id}] Error processing job {uuid}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    try:
        if settings.DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode is enabled.")

        wokers = settings.WORKERS

        for i in range(wokers):
            thread = threading.Thread(target=main, args=(i,))
            thread.start()
    except KeyboardInterrupt:
        print("")
        logger.info("Transcription service stopped.")
