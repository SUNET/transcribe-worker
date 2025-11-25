import json
import logging
import os
import subprocess
import torch
import uuid

from pathlib import Path
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq
from transformers import AutoProcessor
from transformers import pipeline
from typing import Optional
from utils.settings import get_settings

settings = get_settings()


def get_torch_device() -> tuple:
    """
    Determine the device to use for model inference.
    """
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
        model_name: Optional[str] = "KB-Lab/kb-whisper-base",
        language: Optional[str] = "sv",
        speakers: Optional[int] = 0,
        hf_token: Optional[str] = None,
        whisper_cpp_path: Optional[str] = settings.WHISPER_CPP_PATH,
        diarization_object: Optional[Pipeline] = None,
        batch_size: Optional[int] = 16,
        chunk_length_s: Optional[int] = 30,
        use_flash_attention: Optional[bool] = True,
    ) -> None:
        """
        Initializes the WhisperAudioTranscriber with enhanced HF capabilities.
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
        self.__batch_size = batch_size
        self.__chunk_length_s = chunk_length_s
        self.__use_flash_attention = use_flash_attention and self.__device == "cuda:0"
        self.__tokens_to_ignore = [
            "<|nospeech|>",
            "<|p>",
            "<|>",
            '"',
            "<|notimestamps|>",
        ]

        if backend == "hf":
            self.__hf_init()

    def __hf_init(self) -> None:
        """
        Enhanced HF initialization with optimization flags.
        """

        model_kwargs = {
            "torch_dtype": self.__torch_dtype,
            "use_safetensors": True,
            "cache_dir": "cache",
        }

        if self.__use_flash_attention:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.__logger.info(
                    "Flash Attention 2 enabled for faster inference")
            except Exception as e:
                self.__logger.warning(f"Flash Attention 2 not available: {e}")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.__model_name,
            **model_kwargs
        )
        self.model.to(self.__device)

        if hasattr(torch, 'compile') and self.__device == "cuda:0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.__logger.info(
                    "Model compiled with torch.compile for better performance")
            except Exception as e:
                self.__logger.warning(f"torch.compile not available: {e}")

        self.processor = AutoProcessor.from_pretrained(self.__model_name)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.__torch_dtype,
            device=self.__device,
            return_timestamps="word",
            chunk_length_s=self.__chunk_length_s,
            batch_size=self.__batch_size,
        )

    def __seconds_to_srt_time(self, seconds) -> str:
        """
        Convert seconds (float or string) to SRT timestamp format (HH:MM:SS,mmm).
        """

        seconds = float(seconds)
        millis = int(round((seconds % 1) * 1000))
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def __run_cmd(self, command: list) -> bool:
        """
        Run a command using subprocess.run.
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
        """
        Parse SRT timestamp to seconds.
        """

        if timestamp_str is None:
            return None

        time_part, ms_part = timestamp_str.split(",")

        if not ms_part:
            ms_part = "0"

        hours, minutes, seconds = map(int, time_part.split(":"))
        total_seconds = hours * 3600 + minutes * \
            60 + seconds + int(ms_part) / 1000.0

        return total_seconds

    def __process_word_timestamps(self, chunks) -> list:
        """
        Process word-level timestamps into optimized subtitle segments.
        """

        segments = []
        current_segment = {
            "words": [],
            "start": None,
            "end": None,
            "text": ""
        }

        max_chars_per_subtitle = 42
        max_duration = 7.0
        min_duration = 1.2

        for chunk in chunks:
            if "timestamp" not in chunk or chunk["timestamp"][0] is None:
                continue

            start, end = chunk["timestamp"]
            text = chunk["text"].strip()

            if not text or text in self.__tokens_to_ignore:
                continue

            if current_segment["start"] is None:
                current_segment["start"] = start

            potential_text = (current_segment["text"] + " " + text).strip()
            potential_duration = end - current_segment["start"]

            if (len(potential_text) > max_chars_per_subtitle or
                    potential_duration > max_duration):

                if current_segment["text"]:
                    if current_segment["end"] - current_segment["start"] < min_duration:
                        current_segment["end"] = current_segment["start"] + \
                            min_duration

                    segments.append({
                        "start": current_segment["start"],
                        "end": current_segment["end"],
                        "text": current_segment["text"],
                        "duration": current_segment["end"] - current_segment["start"]
                    })

                current_segment = {
                    "words": [text],
                    "start": start,
                    "end": end,
                    "text": text
                }
            else:
                current_segment["words"].append(text)
                current_segment["text"] = potential_text
                current_segment["end"] = end

        if current_segment["text"]:
            if current_segment["end"] - current_segment["start"] < min_duration:
                current_segment["end"] = current_segment["start"] + \
                    min_duration

            segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "text": current_segment["text"],
                "duration": current_segment["end"] - current_segment["start"]
            })

        return segments

    def __process_transcription(self, items, source: str) -> dict:
        """
        Normalize and process transcription items from either HF or whisper.cpp.
        """

        full_transcription = ""
        chunks = []

        for item in items:
            text = item.get("text", "").strip()

            if not text or text in self.__tokens_to_ignore:
                continue

            if source == "cpp":
                try:
                    text = bytes(text, "iso-8859-1").decode("utf-8")
                except UnicodeDecodeError:
                    self.__logger.error(
                        f"Failed to decode {text} from transcription"
                    )
                    continue

            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "

            full_transcription += text

            if source == "hf":
                start, end = item["timestamp"]
                if start is None or end is None:
                    continue

                start_ms = self.__seconds_to_srt_time(str(start))
                end_ms = self.__seconds_to_srt_time(str(end))
                ts_ms = (start_ms, end_ms)
            else:
                if item["tokens"][0]["text"] == "[_BEG_]":
                    start_time_token = item["tokens"][1]["timestamps"]["from"]
                else:
                    start_time_token = item["tokens"][0]["timestamps"]["from"]

                end_time_token = item["tokens"][-1]["timestamps"]["to"]
                ts_ms = (start_time_token, end_time_token)
                start = self.__parse_timestamp(start_time_token)
                end = self.__parse_timestamp(end_time_token)

            chunks.append({
                "timestamp": (start, end),
                "timestamp_ms": ts_ms,
                "text": text,
            })

        segments = self.__process_word_timestamps(chunks)

        converted = {
            "full_transcription": full_transcription.strip(),
            "segments": segments,
            "chunks": chunks,
            "speaker_count": 1,
        }

        self.__result = converted
        self.__transcribed_seconds = segments[-1]["end"] if segments else 0

        return converted

    def __transcribe_hf(self, filepath: str) -> dict:
        """
        Enhanced HF transcription with better parameters.
        """

        result = self.pipe(
            filepath,
            generate_kwargs={
                "task": "transcribe",
                "language": self.__language,
                "temperature": 0.0,  # More deterministic
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
            },
        )

        return self.__process_transcription(result.get("chunks", []), source="hf")

    def __transcribe_cpp(self, filepath: str) -> dict:
        """
        Whisper.cpp transcription (legacy).
        """

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
            "-sns",
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

        match self.__backend:
            case "hf":
                self.__transcribe_hf(self.__audio_path)
            case "cpp":
                self.__transcribe_cpp(self.__audio_path)
            case _:
                raise ValueError(f"Unsupported backend: {self.__backend}")

        if not self.__result:
            raise Exception("Transcription result is not available.")

        return self.__transcribed_seconds

    def diarization(self) -> dict:
        """
        Enhanced speaker diarization with better alignment.
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
            diarization_params = {
                "min_speakers": self.__speakers if self.__speakers > 0 else None,
                "max_speakers": self.__speakers if self.__speakers > 0 else None,
            }

            diarization = self.__diarization_pipeline(
                self.__audio_path,
                **{k: v for k, v in diarization_params.items() if v is not None}
            )

            aligned_segments = self.__align_speakers_enhanced(
                self.__result["segments"], diarization
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

    def __align_speakers_enhanced(self, segments, diarization) -> list:
        """
        Enhanced speaker alignment using segment-level analysis.
        """

        aligned_segments = []

        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            speaker_times = {}

            for seg, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(start, seg.start)
                overlap_end = min(end, seg.end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    speaker_times[speaker] = speaker_times.get(
                        speaker, 0) + overlap_duration

            dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[
                0] if speaker_times else "UNKNOWN"

            aligned_segments.append({
                "start": start,
                "end": end,
                "text": text.strip(),
                "speaker": dominant_speaker,
                "active_speakers": list(speaker_times.keys()),
                "duration": end - start,
            })

        return aligned_segments

    def __get_speaker(self, diarization, time_point) -> str:
        """
        Get the speaker label for a specific time point.
        """

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= time_point <= segment.end:
                return speaker

        return "UNKNOWN"

    def __get_speakers_in_range(self, diarization, start_time, end_time) -> list:
        """
        Get active speakers within a time range.
        """

        active_speakers = set()

        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if not (segment.end < start_time or segment.start > end_time):
                active_speakers.add(speaker)

        return list(active_speakers)

    def subtitles(self) -> str:
        """
        Generate optimized SRT subtitles from segments.
        """

        if not self.__result or "segments" not in self.__result:
            raise Exception(
                "Transcription result is not available or does not contain segments."
            )

        subtitles = ""

        for index, segment in enumerate(self.__result["segments"]):
            start_ms = self.__seconds_to_srt_time(segment["start"])
            end_ms = self.__seconds_to_srt_time(segment["end"])
            text = segment["text"].strip()

            if not text:
                continue

            caption = self.__caption_split_smart(text)

            subtitles += f"{index + 1}\n"
            subtitles += f"{start_ms} --> {end_ms}\n"
            subtitles += f"{caption}\n\n"

        return subtitles

    def __caption_split_smart(self, caption: str) -> str:
        """
        Intelligently split captions at natural break points.
        """

        max_line_length = 42

        if len(caption) <= max_line_length:
            return caption

        mid_point = len(caption) // 2
        search_range = 15

        for offset in range(search_range):
            for pos in [mid_point + offset, mid_point - offset]:
                if 0 < pos < len(caption):
                    char = caption[pos]

                    if char in ",.;:!?-":
                        first_line = caption[:pos + 1].strip()
                        second_line = caption[pos + 1:].strip()
                        return f"{first_line}\n{second_line}"

        pos = mid_point

        while pos > 0 and caption[pos] != " ":
            pos -= 1

        if pos == 0:
            pos = mid_point

        first_line = caption[:pos].strip()
        second_line = caption[pos:].strip()

        return f"{first_line}\n{second_line}"

    def __format_timestamp(self, seconds) -> str:
        """
        Format timestamp in HH:MM:SS format.
        """

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
