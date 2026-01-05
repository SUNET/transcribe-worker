import ffmpeg
import logging
import os
import requests
import tempfile

from enum import Enum
from pathlib import Path
from typing import Optional
from utils import settings
from utils.whisper import WhisperAudioTranscriber

settings = settings.get_settings()


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


class TranscriptionJob:
    def __init__(
        self,
        logger: logging.Logger,
        api_url: str,
        api_file_storage_dir: str,
        hf_whisper: Optional[bool] = False,
        hf_token: Optional[str] = None,
        diarization_object: Optional[object] = None,
    ):
        self.logger = logger
        self.api_url = api_url
        self.api_file_storage_dir = api_file_storage_dir
        self.hf_whisper = hf_whisper
        self.hf_token = hf_token
        self.speakers = 0
        self.diarization_object = diarization_object

    def __enter__(self) -> "TranscriptionJob":
        """
        Initialize the transcription job.
        """
        self.uuid = None
        self.user_id = None
        self.language = None
        self.model_type = None
        self.model = None
        self.filename = None
        self.speakers = 0

        # Ensure the file storage directory exists
        Path(self.api_file_storage_dir).mkdir(parents=True, exist_ok=True)

        return self

    def __exit__(self, *args: object) -> None:
        """
        Cleanup resources when the job is done.
        """
        self.__cleanup()

    def start(self) -> bool:
        """
        Start the transcription job.
        """

        job = self.__get_job()
        if not job:
            return

        self.uuid = job.get("uuid")
        self.user_id = job.get("user_id")
        self.language = job.get("language")
        self.model_type = job.get("model_type")
        self.model = self.__get_model()
        self.speakers = job.get("speakers", 0)
        self.filename = self.uuid
        self.output_format = job.get("output_format", "txt")

        if not self.speakers:
            self.speakers = 0

        self.logger.info(f"Starting transcription job {self.uuid}")
        self.logger.info(f"  HF: {self.hf_whisper}")
        self.logger.info(f"  User: {self.user_id}")
        self.logger.info(f"  Language: {self.language}")
        self.logger.info(f"  Model: {self.model}")
        self.logger.info(f"  Model type: {self.model_type}")
        self.logger.info(f"  Filename: {self.filename}")
        self.logger.info(f"  Speakers: {self.speakers}")
        self.logger.info(f"  Output format: {self.output_format}")

        self.logger.debug("Updating job status to IN_PROGRESS")
        self.__put_status(
            JobStatusEnum.IN_PROGRESS, error=None, transcribed_seconds=None
        )

        self.logger.debug("Fetching file from API broker")
        if not self.__get_file():
            self.logger.error("File download failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="File download failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.debug("Transcoding file")
        if not self.__transcode_file():
            self.logger.error("Transcoding failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Transcoding failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.debug("Transcribing file")
        transcribed_seconds = self.__transcribe()

        if not transcribed_seconds:
            self.logger.error("Transcription failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Transcription failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.debug("Downscaling file")
        if not self.__downscale_file():
            self.logger.error("Downscaling failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="Downscaling failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.debug("Uploading results to backend")
        if not self.__put_result():
            self.logger.error("File upload failed")
            self.__put_status(
                JobStatusEnum.FAILED,
                error="File upload failed",
                transcribed_seconds=None,
            )
            return False

        self.logger.info(f"Job {self.uuid} completed successfully")
        self.__put_status(
            JobStatusEnum.COMPLETED, error=None, transcribed_seconds=transcribed_seconds
        )
        self.logger.info(
            f"Transcription completed, total transcribed seconds: {transcribed_seconds}"
        )

        return True

    def __transcribe(self) -> bool:
        """
        Transcribe the audio file using Hugging Face Whisper.
        """
        self.logger.info("Starting transcription")
        transcriber = WhisperAudioTranscriber(
            self.logger,
            "hf" if self.hf_whisper else "cpp",
            self.__transcoded_data,
            model_name=self.model,
            language=self.language,
            speakers=self.speakers,
            hf_token=self.hf_token,
            diarization_object=self.diarization_object,
        )

        transcribed_seconds = transcriber.transcribe()

        if transcribed_seconds is None:
            return None

        self.__srt = transcriber.subtitles()

        if self.output_format == "txt":
            self.__drz = transcriber.diarization()
        else:
            self.__drz = None

        return transcribed_seconds

    def __downscale_file(self) -> bool:
        """
        Downscale videos to a smaller size.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(self.__downloaded_data)
            tmp_path = tmp.name

            self.__downscaled_file = tempfile.gettempdir() + "/" + self.uuid + ".mp4"

            try:
                stdout, stderr = (
                    ffmpeg.input(tmp_path)
                    .filter(
                        "scale",
                        -2,
                        360,
                        flags="lanczos",
                    )
                    .output(
                        self.__downscaled_file,
                        format="mp4",
                        vcodec="libx264",
                        preset="veryfast",
                        crf=22,
                        **{
                            "profile:v": "high",
                            "level": "3.1",
                            "pix_fmt": "yuv420p",
                            "g": 48,
                            "keyint_min": 48,
                            "sc_threshold": 0,
                            "c:a": "aac",
                            "b:a": "128k",
                            "ar": 48000,
                            "ac": 2,
                            "movflags": "+faststart",
                        },
                    )
                ).run(capture_stdout=True, capture_stderr=True)

            except Exception as e:
                self.logger.error(f"FFmpeg error during downscaling: {e}")
                os.unlink(self.__downscaled_file)
                return False
            finally:
                os.unlink(tmp_path)

        return True

    def __transcode_file(self) -> bool:
        """
        Transcode the audio file using ffmpeg.
        The transcoded format should be 16kHz mono WAV.
        """

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            tmp.write(self.__downloaded_data)
            tmp_path = tmp.name

            stdout, stderr = (
                ffmpeg.input(tmp_path)
                .output("pipe:1", format="wav", acodec="pcm_s16le", ac=1, ar="16k")
                .run(capture_stdout=True, capture_stderr=True)
            )

            self.__transcoded_data = stdout

        return True

    def __get_job(self) -> dict:
        """
        Get the next job from the API broker.
        """
        try:
            response = requests.get(
                f"{self.api_url}/next",
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()
            job = response.json()["result"]
            if "status" in job and job["status"] != JobStatusEnum.IN_PROGRESS:
                self.logger.info(f"Job {job['uuid']} is not in_progress. Skipping.")
                return {}

        except Exception as e:
            self.logger.error(f"Error fetching next job: {e}")
            return {}

        return job

    def __get_file(self) -> bool:
        """
        Download the file from the API broker.
        """

        try:
            response = requests.get(
                f"{self.api_url}/{self.user_id}/{self.uuid}/file",
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()

            if response.status_code != 200:
                self.logger.error(f"Error downloading file: {response.status_code}")
                raise Exception("File not downloaded")

            self.__downloaded_data = response.content

        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False

        return True

    def __put_status(
        self, status: JobStatusEnum, error: str, transcribed_seconds: int
    ) -> bool:
        """
        Update the job status in the API broker.
        """

        try:
            response = requests.put(
                f"{self.api_url}/{self.uuid}",
                json={
                    "status": status,
                    "error": error,
                    "transcribed_seconds": transcribed_seconds,
                },
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error updating job status: {e}")
            return False

        return True

    def __upload_mp4(self) -> bool:
        """
        Upload the MP4 file to the API broker.
        """

        try:
            response = requests.put(
                f"{self.api_url}/{self.user_id}/{self.uuid}/file",
                files={"file": open(self.__downscaled_file, "rb")},
                cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
            )
            os.unlink(self.__downscaled_file)

            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error uploading MP4 file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error uploading MP4 file: {e}")
            return False

    def __put_result(self) -> int:
        """
        Upload the file to the API broker.
        """
        header = {
            "Content-Type": "application/json",
        }

        for output_format in ["srt", "json", "mp4"]:
            try:
                if output_format == "mp4":
                    self.__upload_mp4()
                    continue

                json_data = {}

                if output_format == "json":
                    if not self.__drz:
                        continue
                    json_data["result"] = self.__drz
                elif output_format == "srt":
                    json_data["result"] = self.__srt

                json_data["format"] = output_format
                response = requests.put(
                    f"{self.api_url}/{self.user_id}/{self.uuid}/result",
                    json=json_data,
                    headers=header,
                    cert=(settings.SSL_CERTFILE, settings.SSL_KEYFILE),
                )
                response.raise_for_status()
            except requests.RequestException as e:
                self.logger.error(f"Error uploading {output_format} file: {e}")
                return False

            self.logger.info(f"Uploaded {output_format} file for job {self.uuid}")

        return True

    def __get_model(self) -> str:
        """
        Return the correct model file based on
        model type and language.
        """

        if self.hf_whisper:
            model = settings.WHISPER_MODELS_HF[self.language][self.model_type.lower()]
        else:
            model = (
                "models/"
                + settings.WHISPER_MODELS_CPP[self.language][self.model_type.lower()]
            )

        return model

    def __cleanup(self) -> bool:
        """
        Delete all files related to the job.
        """

        if not self.uuid:
            return

        file_path = Path(self.api_file_storage_dir) / self.uuid
        if file_path.exists():
            file_path.unlink()
            self.logger.info(f"Deleted file {file_path}")

        for file in Path(self.api_file_storage_dir).glob(f"{self.uuid}.*"):
            if file.exists():
                file.unlink()
                self.logger.info(f"Deleted file {file}")

        return True
