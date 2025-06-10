import logging
import threading

from random import randint
from time import sleep
from utils.job import TranscriptionJob
from utils.log import get_logger
from utils.settings import get_settings

settings = get_settings()
logger = get_logger()


def main(worker_id: int):
    """
    Main function to fetch jobs and process them.
    """

    api_broker_url = settings.API_BACKEND_URL
    api_file_storage_dir = settings.API_FILE_STORAGE_DIR
    api_version = settings.API_VERSION
    api_url = f"{api_broker_url}/api/{api_version}/job"
    api_token = settings.OIDC_TOKEN

    logger.info(f"[{worker_id}] Starting transcription service, server URL: {api_url}")

    while True:
        sleep_time = randint(10, 60)
        logger.debug(f"[{worker_id}] Fetching next job in {sleep_time} seconds.")

        sleep(sleep_time)

        with TranscriptionJob(
            logger,
            api_url,
            api_token,
            api_file_storage_dir,
            hf_whisper=settings.HF_WHISPER,
            hf_token=settings.HF_TOKEN,
        ) as job:
            job.start()


if __name__ == "__main__":
    try:
        if settings.API_DEBUG:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode is enabled.")

        workers = settings.API_WORKERS

        for i in range(workers):
            thread = threading.Thread(target=main, args=(i,))
            thread.start()
    except KeyboardInterrupt:
        print("")
        logger.info("Transcription service stopped.")
