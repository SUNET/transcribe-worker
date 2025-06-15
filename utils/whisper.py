import os
import json
import uuid
import torch
import subprocess

from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq
from transformers import AutoProcessor
from transformers import pipeline
from typing import Optional
from pathlib import Path
from utils.settings import get_settings

settings = get_settings()


class WhisperAudioTranscriber:
    def __init__(
        self,
        backend: str,
        audio_path: str,
        model_name: Optional[str] = "KBLab/kb-whisper-base",
        language: Optional[str] = "sv",
        hf_token: Optional[str] = None,
        whisper_cpp_path: Optional[str] = "whisper.cpp",
    ):
        """
        Initializes the WhisperAudioTranscriber with the audio
        file path, model name,
        """

        self.__audio_path = audio_path
        self.__model_name = model_name
        self.__hf_token = hf_token
        self.__device, self.__torch_dtype = self.__get_device(torch)
        self.__language = language
        self.__result = None
        self.__whisper_cpp_path = whisper_cpp_path
        self.__backend = backend

        if backend == "hf":
            self.__hf_init()

    def __hf_init(self):
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

    def __diarization_init(self):
        """
        Initializes the diarization pipeline using HuggingFace's PyAnnote.
        """

        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.__hf_token
        )
        self.diarization_pipeline.to(torch.device(self.__device))

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
            print(f"Running command: {command_str}")
            result = subprocess.run(command, capture_output=True)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=command_str,
                    output=result.stdout.decode(),
                    stderr=result.stderr.decode(),
                )
        except Exception:
            return None

        return True

    def __parse_timestamp(self, timestamp_str):
        # Split by comma to separate seconds and milliseconds
        time_part, ms_part = timestamp_str.split(",")

        # Split time part into hours, minutes, seconds
        hours, minutes, seconds = map(int, time_part.split(":"))

        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + int(ms_part) / 1000.0

        return total_seconds

    def __transcribe_cpp(self, filepath: str):
        """
        Transcribe the audio file using whisper.cpp, we expect the executable
        to be in PATH.
        """
        temp_filename = str(uuid.uuid4())
        command = [
            self.__whisper_cpp_path,
            "-l",
            self.__language,
            "-ojf",
            "-of",
            settings.API_FILE_STORAGE_DIR + "/" + temp_filename,
            "-m",
            self.__model_name,
            "-f",
            filepath,
        ]

        if not self.__run_cmd(command):
            raise Exception(f"Failed to run whisper.cpp command: {' '.join(command)}")

        with open(
            str(Path(settings.API_FILE_STORAGE_DIR) / f"{temp_filename}.json"), "rb"
        ) as f:
            json_str = f.read()
            json_str = json_str.decode("latin-1")
            result = json.loads(json_str)

        full_transcription = ""
        segments = []
        chunks = []

        for item in result.get("transcription", []):
            # Add text to full transcription
            text = item.get("text", "").strip()
            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "
            full_transcription += text

            # Convert timestamps from "HH:MM:SS,mmm" format to seconds
            start_time = self.__parse_timestamp(item["timestamps"]["from"])
            end_time = self.__parse_timestamp(item["timestamps"]["to"])
            duration = end_time - start_time

            # Create segment in new format
            segment = {
                "start": start_time,
                "end": end_time,
                "text": text,
                "speaker": "SPEAKER_01",  # Default speaker - could be enhanced with speaker detection
                "active_speakers": ["SPEAKER_01"],
                "duration": duration,
            }

            chunk = {
                "timestamp": (start_time, end_time),
                "text": text,
            }

            segments.append(segment)
            chunks.append(chunk)

        # Create the converted format
        converted = {
            "full_transcription": full_transcription,
            "segments": segments,
            "chunks": chunks,
            "speaker_count": 1,  # Default to 1 - could be enhanced with actual speaker detection
        }

        print(f"Transcription result: {json.dumps(converted, indent=2)}")

        self.__result = converted

        return converted

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

        return self.__result

    def diarization(self) -> dict:
        """
        Perform speaker diarization on the transcribed audio.
        """
        self.__diarization_init()

        if not self.diarization_pipeline:
            raise Exception(
                "Diarization pipeline not initialized. Please provide a HuggingFace token."
            )

        if not self.__result:
            raise Exception(
                "Transcription result is not available. Please transcribe first."
            )

        try:
            diarization = self.diarization_pipeline(self.__audio_path)
            aligned_segments = self.__align_speakers(
                self.__result["chunks"], diarization
            )

            return {
                "full_transcription": self.__result["full_transcription"],
                "segments": aligned_segments,
                "speaker_count": len(list(diarization.labels())),
            }

        except Exception as e:
            print(f"Error during transcription with diarization: {str(e)}")
            return None

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
        for chunk in self.__result["chunks"]:
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()
            if not text:
                continue

            caption = self.__caption_split(text)
            subtitles += f"{index + 1}\n"
            subtitles += f" {self.__format_timestamp(start)} --> {self.__format_timestamp(end)}\n"
            subtitles += f"{caption}\n\n"

            index += 1

        return subtitles

    def __get_device(self, torch: torch):
        """
        Determine the device to use for model inference.
        """
        if torch.cuda.is_available():
            return "cuda:0", torch.float16
        elif torch.backends.mps.is_available():
            return "mps", torch.float16
        else:
            return "cpu", torch.float32

    def __align_speakers(self, transcription_chunks, diarization):
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

    def __get_speaker(self, diarization, time_point):
        """
        Get the speaker label for a specific time point in the diarization.
        """
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= time_point <= segment.end:
                return speaker

        return "UNKNOWN"

    def __get_speakers_in_range(self, diarization, start_time, end_time):
        """
        Get a list of active speakers within a specific time range in the diarization.
        """
        active_speakers = set()

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if not (segment.end < start_time or segment.start > end_time):
                active_speakers.add(speaker)

        return list(active_speakers)

    def __format_timestamp(self, seconds):
        """
        Format a timestamp in seconds to MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __caption_split(self, caption):
        """
        Split a caption into two parts if it exceeds a certain length.
        """
        if len(caption) < 42:
            return f" {caption}"

        words_list = caption.split()
        words_len = len(words_list)
        word_mid = words_len // 2

        left_part = words_list[:word_mid]
        right_part = words_list[word_mid:]

        new_caption = " "
        new_caption += " ".join(left_part) + "\n"
        new_caption += " "
        new_caption += " ".join(right_part)

        return new_caption
