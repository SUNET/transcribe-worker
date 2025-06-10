import requests
import subprocess
import logging
import json

from enum import Enum
from pathlib import Path
from typing import Optional
from utils.whisper import WhisperAudioTranscriber


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
        api_token: str,
        api_file_storage_dir: str,
        hf_whisper: Optional[bool] = False,
        hf_token: Optional[str] = None,
    ):
        self.logger = logger
        self.api_url = api_url
        self.api_token = api_token
        self.api_file_storage_dir = api_file_storage_dir
        self.hf_whisper = hf_whisper
        self.hf_token = hf_token

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
        self.filename = self.uuid

        self.logger.info(f"Starting transcription job {self.uuid}")
        self.logger.info(f"  HF: {self.hf_whisper}")
        self.logger.info(f"  User: {self.user_id}")
        self.logger.info(f"  Language: {self.language}")
        self.logger.info(f"  Model: {self.model}")
        self.logger.info(f"  Model type: {self.model_type}")
        self.logger.info(f"  Filename: {self.filename}")

        self.__put_status(JobStatusEnum.IN_PROGRESS, error=None)

        if not self.__get_file():
            self.__put_status(JobStatusEnum.FAILED, error="File download failed")
            return False

        if not self.__transcode_file():
            self.__put_status(JobStatusEnum.FAILED, error="Transcoding failed")
            return False

        if self.hf_whisper:
            res = self.__transcribe_hf()
        else:
            res = self.__transcribe_file()

        if not res:
            self.__put_status(JobStatusEnum.FAILED, error="Transcription failed")
            return False

        if not self.__downscale_file():
            self.__put_status(JobStatusEnum.FAILED, error="Downscaling failed")
            return False

        if not self.__put_file():
            self.__put_status(JobStatusEnum.FAILED, error="File upload failed")
            return False

        self.logger.info(f"Job {self.uuid} completed successfully")
        self.__put_status(JobStatusEnum.COMPLETED, error=None)

        return True

    def __transcribe_hf(self) -> bool:
        """
        Transcribe the audio file using Hugging Face Whisper.
        """
        self.logger.info("Starting transcription with Hugging Face Whisper")
        try:
            transcriber = WhisperAudioTranscriber(
                audio_path=str(
                    Path(self.api_file_storage_dir) / f"{self.filename}.wav"
                ),
                model_name=self.model,
                language=self.language,
                hf_token=self.hf_token,
            )

            transcriber.transcribe()
            srt = transcriber.subtitles()
            drz = transcriber.diarization()

            with open(
                Path(self.api_file_storage_dir) / f"{self.filename}.srt", "w"
            ) as srt_file:
                srt_file.write(srt)

            with open(
                Path(self.api_file_storage_dir) / f"{self.filename}.json", "w"
            ) as json_file:
                json_file.write(json.dumps(dict(drz)))

        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return False

        return True

    def __run_cmd(self, command: list) -> bool:
        """
        Run a command using subprocess.run.
        Raises an exception if the command fails.
        """
        try:
            command_str = " ".join(command)
            self.logger.debug(f"Executing command: {command_str}")
            result = subprocess.run(command, capture_output=True)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=command_str,
                    output=result.stdout.decode(),
                    stderr=result.stderr.decode(),
                )
        except Exception as e:
            self.logger.error(f"Error when executing command: {e}")

        return True

    def __downscale_file(self) -> bool:
        """
        Downscale videos to a smaller size.
        """

        output_filename = f"{self.filename}.mp4"
        command = [
            "ffmpeg",
            "-i",
            str(Path(self.api_file_storage_dir) / self.filename),
            "-vf",
            "scale=-2:320",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(Path(self.api_file_storage_dir) / output_filename),
            "-y",
        ]

        try:
            self.__run_cmd(command)
        except Exception as e:
            self.logger.error(f"Error during downscaling: {e}")
            return False

        return True

    def __transcode_file(self) -> bool:
        """
        Transcode the audio file using ffmpeg.
        The transcoded format should be 16kHz mono WAV.
        """

        output_filename = f"{self.filename}.wav"
        command = [
            "ffmpeg",
            "-i",
            str(Path(self.api_file_storage_dir) / self.filename),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            "-y",
            str(Path(self.api_file_storage_dir) / output_filename),
            "-y",
        ]

        try:
            self.__run_cmd(command)
        except Exception as e:
            self.logger.error(f"Error during transcoding: {e}")
            return False

        return True

    def __transcribe_file(self) -> bool:
        """
        Transcribe the audio file using whisper.cpp, we expect the executable
        to be in PATH.
        """
        command = [
            "whisper.cpp",
            "-l",
            self.language,
            "-osrt",
            "-ojf",
            "-otxt",
            "-ovtt",
            "-of",
            str(Path(self.api_file_storage_dir) / self.filename),
            "-m",
            self.model,
            "-f",
            str(Path(self.api_file_storage_dir) / f"{self.filename}.wav"),
        ]

        try:
            self.__run_cmd(command)
        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return False
        else:
            self.logger.info(f"Transcription completed: {self.filename}")

        return True

    def __get_job(self) -> dict:
        """
        Get the next job from the API broker.
        """
        header = {"Authorization": f"Bearer {self.api_token}"}

        try:
            response = requests.get(f"{self.api_url}/next", headers=header)
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
        header = {"Authorization": f"Bearer {self.api_token}"}

        try:
            response = requests.get(
                f"{self.api_url}/{self.user_id}/{self.uuid}/file", headers=header
            )
            response.raise_for_status()

            if response.status_code != 200:
                self.logger.error(f"Error downloading file: {response.status_code}")
                raise Exception("File not downloaded")

            file_path = Path(self.api_file_storage_dir) / self.uuid

            with open(file_path, "wb") as f:
                f.write(response.content)

        except Exception as e:
            self.logger.error(f"Error downloading file: {e}")
            return False

        return True

    def __put_status(self, status: JobStatusEnum, error: str) -> bool:
        """
        Update the job status in the API broker.
        """
        header = {"Authorization": f"Bearer {self.api_token}"}

        try:
            response = requests.put(
                f"{self.api_url}/{self.uuid}",
                json={"status": status, "error": error},
                headers=header,
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Error updating job status: {e}")
            return False

        return True

    def __put_file(self) -> bool:
        """
        Upload the file to the API broker.
        """
        header = {"Authorization": f"Bearer {self.api_token}"}

        try:
            for output_format in ["srt", "vtt", "json", "txt", "mp4"]:
                file_path = (
                    Path(self.api_file_storage_dir) / f"{self.uuid}.{output_format}"
                )

                if not file_path.exists():
                    self.logger.debug(
                        f"File {file_path} does not exist, not uploading."
                    )
                    continue

                with open(file_path, "rb") as fd:
                    response = requests.put(
                        f"{self.api_url}/{self.user_id}/{self.uuid}/result",
                        files={"file": fd},
                        headers=header,
                    )
                    response.raise_for_status()

                self.logger.info(f"Uploaded {output_format} file for job {self.uuid}")
        except Exception as e:
            self.logger.error(f"Error uploading file: {e}")
            return False

        return True

    def __get_model(self) -> str:
        """
        Return the correct model file based on
        model type and language.

        If langauge = sv then use kb-whisper else
        use the default whisper model.
        """

        if self.hf_whisper and self.language == "sv":
            match self.model_type:
                case "tiny":
                    model = "KBLab/kb-whisper-tiny"
                case "base":
                    model = "KBLab/kb-whisper-base"
                case "large":
                    model = "KBLab/kb-whisper-large"
        elif self.hf_whisper and self.language == "en":
            match self.model_type:
                case "tiny":
                    model = "openai/whisper-tiny"
                case "base":
                    model = "openai/whisper-base"
                case "large":
                    model = "openai/whisper-large"
        else:
            model = "models/"
            model += "sv" if self.language == "sv" else "en"

            match self.model_type:
                case "tiny":
                    model += "_tiny"
                case "base":
                    model += "_base"
                case "large":
                    model += "_large"
                case _:
                    model += "_base"

            model += ".bin"

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

        for ext in [".txt", ".vtt", ".json", "mp4"]:
            file_path = Path(self.api_file_storage_dir) / f"{self.uuid}{ext}"
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file {file_path}")

        return True
