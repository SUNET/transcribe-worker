import logging
import os
import shutil
import torch

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


def models_download(logger: logging.Logger) -> None:
    """
    Preload models to cache.
    """

    logger.info("Removing cache...")
    shutil.rmtree("cache", ignore_errors=True)
    device, torch_dtype = get_torch_device()

    for item in settings.WHISPER_MODELS_HF.items():
        langugage = item[0]

        for model in item[1].values():
            logger.info(f"Downloading model for {langugage}: {model}...")
            _ = AutoModelForSpeechSeq2Seq.from_pretrained(
                model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                cache_dir="cache",
                token=settings.HF_TOKEN,
            )


class WhisperAudioTranscriber:
    def __init__(
        self,
        logger: logging.Logger,
        audio_path: str,
        model_name: Optional[str] = "KBLab/kb-whisper-base",
        language: Optional[str] = "sv",
        speakers: Optional[int] = 0,
        hf_token: Optional[str] = None,
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
        self.__logger = logger
        self.__speakers = speakers
        self.__diarization_pipeline = diarization_object
        self.__hf_init()

    def __hf_init(self) -> None:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.__model_name,
            torch_dtype=self.__torch_dtype,
            use_safetensors=True,
            cache_dir="cache",
            token=self.__hf_token,
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
            chunk_length_s=30,
            stride_length_s=(4, 2),
        )

    def __seconds_to_srt_time(self, seconds) -> str:
        """
        Convert seconds (float or string) HH:MM:SS,mmm.
        """

        seconds = float(seconds)
        millis = int(round((seconds % 1) * 1000))
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def __parse_timestamp(self, timestamp_str) -> Optional[float]:
        if timestamp_str is None:
            return None

        time_part, ms_part = timestamp_str.split(",")

        if not ms_part:
            ms_part = "0"

        hours, minutes, seconds = map(int, time_part.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds + int(ms_part) / 1000.0

        return total_seconds

    def __process_transcription(self, items) -> dict:
        """
        Normalize and process transcription items.
        """

        full_transcription = ""
        segments = []
        chunks = []

        for index, item in enumerate(items):
            text = item["text"].strip()
            start, end = item["timestamp"]

            if not start or not end:
                continue

            if full_transcription and not full_transcription.endswith(" "):
                full_transcription += " "

            full_transcription += text

            segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "duration": end - start,
                }
            )

            chunks.append(
                {
                    "timestamp": (start, end),
                    "timestamp_ms": (
                        self.__seconds_to_srt_time(start),
                        self.__seconds_to_srt_time(end),
                    ),
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

    def transcribe(self) -> dict:
        """
        Transcribe the audio file and return the transcription result.
        """
        if not os.path.exists(self.__audio_path):
            raise FileNotFoundError(f"Audio file {self.__audio_path} does not exist.")

        result = self.pipe(
            self.__audio_path,
            generate_kwargs={"task": "transcribe", "language": self.__language},
        )

        self.__process_transcription(result.get("chunks", []))

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

    def __timestamp_to_float(self, timestamp: str) -> float:
        """
        Convert a timestamp string in the format HH:MM:SS,mmm to float seconds.
        """
        time_part, ms_part = timestamp.split(",")

        if not ms_part:
            ms_part = "0"

        hours, minutes, seconds = map(int, time_part.split(":"))
        total_seconds = hours * 3600 + minutes * 60 + seconds + int(ms_part) / 1000.0

        return total_seconds

    def subtitles(self) -> str:
        """
        Generate SRT subtitles following accessibility rules.
        """
        if not self.__result or "chunks" not in self.__result:
            raise Exception(
                "Transcription result is not available or does not contain chunks."
            )

        subtitles = ""

        for index, segment in enumerate(self.__result["segments"]):
            start_time = self.__seconds_to_srt_time(segment["start"])
            end_time = self.__seconds_to_srt_time(segment["end"])
            text = segment["text"].strip()

            subtitles += f"{index + 1}\n"
            subtitles += f"{start_time} --> {end_time}\n"
            subtitles += f"{text}\n\n"

        return subtitles


if __name__ == "__main__":
    w = WhisperAudioTranscriber(
        logger=logging.getLogger(),
        audio_path="test.wav",
        model_name="openai/whisper-base",
        language="en",
        speakers=2,
        hf_token=settings.HF_TOKEN,
    )

    w.transcribe()
    print(w.subtitles())
