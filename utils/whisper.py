import json
import logging
import os
import subprocess
import torch
import uuid

from pathlib import Path
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional
from utils.settings import get_settings

settings = get_settings()


def get_torch_device() -> tuple:
    """
    Determine the device to use for model inference.
    """
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    elif torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def diarization_init(hf_token: str) -> Optional[Pipeline]:
    """
    Initializes the diarization pipeline using HuggingFace's PyAnnote.
    """
    device, _ = get_torch_device()

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    ).to(torch.device(device))


class WhisperAudioTranscriber:
    def __init__(
        self,
        logger: logging.Logger,
        backend: str,
        audio_path: str,
        model_name: Optional[str] = "KBLab/kb-whisper-base",
        language: Optional[str] = "sv",
        speakers: Optional[int] = 0,
        hf_token: Optional[str] = None,
        whisper_cpp_path: Optional[str] = settings.WHISPER_CPP_PATH,
        diarization_object: Optional[Pipeline] = None,
    ) -> None:
        """
        Initializes the WhisperAudioTranscriber with the audio
        file path, model name,
        """

        self.__audio_path = audio_path
        self.__model_name = model_name
        self.__hf_token = hf_token
        self.__device, self.__torch_dtype = get_torch_device()
        self.__language = language
        self.__result = None
        self.__whisper_cpp_path = whisper_cpp_path
        self.__backend = backend
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__tokens_to_ignore = [
            "<|nospeech|>",
            "<|p>",
            "<|>",
            '"',
        ]

        if backend == "hf":
            self.__hf_init()

    def __hf_init(self) -> None:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.__model_name,
            torch_dtype=self.__torch_dtype,
            use_safetensors=True,
            cache_dir="cache",
        )
        self.model.to(self.__device)
        self.processor = AutoProcessor.from_pretrained(self.__model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.__model_name,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.__torch_dtype,
            device=self.__device,
            return_timestamps=True,
        )

    def __seconds_to_srt_time(self, seconds) -> str:
        """
        Convert seconds (float or string) to SRT timestamp format
        (HH:MM:SS,mmm).
        """
        seconds = float(seconds)  # ensure it's a float
        millis = int(round((seconds % 1) * 1000))
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def __transcribe_hf(self, filepath: str) -> list:
        """
        Transcribe the audio file using the Whisper model.
        """
        self.__result = self.pipe(
            filepath,
            generate_kwargs={"task": "transcribe", "language": self.__language},
        )

        return self.__result

    def __run_cmd(self, command: list) -> bool:
        """
        Run a command using subprocess.run.
        Raises an exception if the command fails.
        """
        try:
            command_str = " ".join(command)
            self.__logger.debug(f"Running command: {command_str}")
            result = subprocess.run(command, capture_output=True)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=command_str,
                    output=result.stdout.decode(),
                    stderr=result.stderr.decode(),
                )
        except Exception as e:
            self.__logger.error(f"Error running command: {e}")
            return None

        return True

    def __parse_timestamp(self, timestamp_str) -> Optional[float]:
        if timestamp_str is None:
            return None

        time_part, ms_part = timestamp_str.split(",")

        if not ms_part:
            ms_part = "0"

        # Split time part into hours, minutes, seconds
        hours, minutes, seconds = map(int, time_part.split(":"))

        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + int(ms_part) / 1000.0

        return total_seconds

    def __process_transcription(self, items, source: str) -> dict:
        """
        Normalize and process transcription items from either HF or
        whisper.cpp.
        """
        full_transcription = ""
        segments = []
        chunks = []

        for index, item in enumerate(items):
            text = item.get("text", "").strip()

            if not text:
                continue

            if text in self.__tokens_to_ignore:
                continue

            if source == "cpp":
                try:
                    text = bytes(text, "iso-8859-1").decode("utf-8")
                except UnicodeDecodeError:
                    self.__logger.error(
                        f"Failed to decode {text} from transcription, using ISO-8859-1 encoding."
                    )
                    continue

            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "

            full_transcription += text

            if source == "hf":
                start, end = item["timestamp"]
                start_ms = self.__seconds_to_srt_time(str(start))
                end_ms = self.__seconds_to_srt_time(str(end))
                start_time = self.__parse_timestamp(start_ms)
                end_time = self.__parse_timestamp(end_ms)
                ts_ms = (start_ms, end_ms)

            else:
                if item["tokens"][0]["text"] == "[_BEG_]":
                    start_time_token = item["tokens"][1]["timestamps"]["from"]
                    start_time = self.__parse_timestamp(start_time_token)
                else:
                    start_time_token = item["tokens"][0]["timestamps"]["from"]
                    start_time = self.__parse_timestamp(start_time_token)

                end_time_token = item["tokens"][-1]["timestamps"]["to"]
                end_time = self.__parse_timestamp(end_time_token)

                if (end_time - start_time) < 1.5:
                    time_to_add = 1.5 - (end_time - start_time)
                    next_item_start_time = self.__parse_timestamp(
                        items[index + 1]["tokens"][0]["timestamps"]["from"]
                        if index + 1 < len(items)
                        else None
                    )

                    end_time += time_to_add

                    if next_item_start_time and end_time > next_item_start_time:
                        end_time = next_item_start_time - 0.1

                    end_time_token = self.__seconds_to_srt_time(str(end_time))

                ts_ms = (start_time_token, end_time_token)

            duration = end_time - start_time

            segments.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "text": text,
                    "duration": duration,
                }
            )

            chunks.append(
                {
                    "timestamp": (start_time, end_time),
                    "timestamp_ms": ts_ms,
                    "text": text,
                }
            )

        converted = {
            "full_transcription": full_transcription,
            "segments": segments,
            "chunks": chunks,
            "speaker_count": 1,
        }

        self.__result = converted
        self.__transcribed_seconds = segments[-1]["end"] if segments else 0

        return converted

    def __transcribe_hf(self, filepath: str) -> dict:
        result = self.pipe(
            filepath,
            generate_kwargs={"task": "transcribe", "language": self.__language},
        )

        breakpoint()

        return self.__process_transcription(result.get("chunks", []), source="hf")

    def __transcribe_cpp(self, filepath: str) -> dict:
        temp_filename = str(uuid.uuid4())
        command = [
            self.__whisper_cpp_path,
            "-l",
            self.__language,
            "-ojf",
            "-of",
            str(Path(settings.FILE_STORAGE_DIR) / temp_filename),
            "-m",
            self.__model_name,
            "-f",
            filepath,
        ]

        if not self.__run_cmd(command):
            raise Exception("Failed to run whisper.cpp command")

        json_path = Path(settings.FILE_STORAGE_DIR) / f"{temp_filename}.json"
        with open(json_path, "rb") as f:
            json_str = f.read()
        os.remove(json_path)

        result = json.loads(json_str.decode("iso-8859-1"))
        return self.__process_transcription(
            result.get("transcription", []), source="cpp"
        )

    def transcribe(self) -> dict:
        """
        Transcribe the audio file and return the transcription result.
        """
        if not os.path.exists(self.__audio_path):
            raise FileNotFoundError(f"Audio file {self.__audio_path} does not exist.")

        if self.__backend == "hf":
            self.__transcribe_hf(self.__audio_path)
        elif self.__backend == "cpp":
            self.__transcribe_cpp(self.__audio_path)
        else:
            raise ValueError(f"Unsupported backend: {self.__backend}")

        if not self.__result:
            raise Exception("Transcription result is not available.")

        return self.__transcribed_seconds

    def diarization(self) -> dict:
        """
        Perform speaker diarization on the transcribed audio.
        """
        if not self.__diarization_pipeline:
            self.__logger.info("Initializing diarization pipeline...")
            self.__diarization_pipeline = diarization_init(self.__hf_token)
        else:
            self.__logger.info("Diarization pipeline already initialized.")

        if not self.__diarization_pipeline:
            raise Exception(
                "Diarization pipeline not initialized. Please provide a HuggingFace token."
            )

        if not self.__result:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        try:
            diarization = self.__diarization_pipeline(
                self.__audio_path, num_speakers=int(self.__speakers)
            )
            aligned_segments = self.__align_speakers(
                self.__result["chunks"], diarization
            )

            return {
                "full_transcription": self.__result["full_transcription"],
                "segments": aligned_segments,
                "speaker_count": len(list(diarization.labels())) if diarization else 0,
            }
        except Exception as e:
            self.__logger.error(
                f"Error during transcription with diarization: {str(e)}"
            )
            return None

    def __align_speakers(self, transcription_chunks, diarization) -> list:
        """
        Align transcription chunks with speaker diarization results.
        """
        aligned_segments = []

        for chunk in transcription_chunks:
            chunk_start = chunk["timestamp"][0]
            chunk_end = chunk["timestamp"][1]
            chunk_text = chunk["text"]

            chunk_middle = (chunk_start + chunk_end) / 2
            dominant_speaker = self.__get_speaker(diarization, chunk_middle)
            active_speakers = self.__get_speakers_in_range(
                diarization, chunk_start, chunk_end
            )

            aligned_segments.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": chunk_text.strip(),
                    "speaker": dominant_speaker,
                    "active_speakers": active_speakers,
                    "duration": chunk_end - chunk_start,
                }
            )

        return aligned_segments

    def __get_speaker(self, diarization, time_point) -> str:
        """
        Get the speaker label for a specific time point in the diarization.
        """
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= time_point <= segment.end:
                return speaker

        return "UNKNOWN"

    def __get_speakers_in_range(self, diarization, start_time, end_time) -> list:
        """
        Get a list of active speakers within a specific time range in the
        diarization.
        """
        active_speakers = set()

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if not (segment.end < start_time or segment.start > end_time):
                active_speakers.add(speaker)

        return list(active_speakers)

    def subtitles(self) -> str:
        """
        Generate subtitles from the transcription result.
        """
        if not self.__result or "chunks" not in self.__result:
            raise Exception(
                "Transcription result is not available or does not contain chunks."
            )

        index = 0
        subtitles = ""

        for index, chunk in enumerate(self.__result["chunks"]):
            start, end = chunk["timestamp_ms"]
            text = chunk["text"].strip()

            if not text:
                continue

            caption = self.__caption_split(text)
            subtitles += f"{index + 1}\n"
            subtitles += f"{start} --> {end}\n"
            subtitles += f"{caption}\n\n"

        return subtitles

    def __format_timestamp(self, seconds) -> str:
        """
        Format a timestamp in seconds to MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __caption_split(self, caption) -> str:
        """
        Split a caption into two parts if it exceeds a certain length.
        """
        if len(caption) < 42:
            return f"{caption}"

        current_position = len(caption) // 2
        characater = caption[current_position]

        while characater != " ":
            characater = caption[current_position]
            current_position -= 1

        first_line = caption[: current_position + 1].strip()
        second_line = caption[current_position + 1 :].strip()
        new_caption = f"{first_line}\n{second_line}"

        return new_caption
