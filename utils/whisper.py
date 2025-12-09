import logging
import os
import shutil
import torch

from pyannote.audio import Pipeline, core
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import List, Optional, Tuple
from utils.settings import get_settings

settings = get_settings()
original_torch_load = torch.load


def __trusted_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


torch.load = __trusted_load


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
    torch.serialization.add_safe_globals([core.task.Specifications])

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
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
            dtype=self.__torch_dtype,
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
            dtype=self.__torch_dtype,
            device=self.__device,
            return_timestamps="word",
        )

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

    def __format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
        """

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def __split_into_lines(self, words: List[dict], max_chars: int = 42) -> List[str]:
        """
        Split words into lines respecting max character limit.
        """

        lines = []
        current_line = []
        current_length = 0

        for word_data in words:
            word = word_data["text"].strip()

            if not word:
                continue

            word_length = len(word)
            space_length = 1 if current_line else 0

            if current_length + space_length + word_length <= max_chars:
                current_line.append(word)
                current_length += space_length + word_length
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def __process_transcription(
        self,
        chunks: List[dict],
        max_chars: int = 42,
        max_lines: int = 2,
        max_duration: float = 7.0,
        min_duration: float = 1.0,
    ) -> List[Tuple[float, float, str]]:
        """
        Create subtitle blocks following accessibility standards.
        """

        i = 0
        subtitles = []

        while i < len(chunks):
            words = []
            start_time = chunks[i]["timestamp"][0]

            while i < len(chunks):
                word = chunks[i]["text"].strip()
                if word:
                    words.append(chunks[i])

                end_time = chunks[i]["timestamp"][1]
                duration = end_time - start_time

                lines = self.__split_into_lines(words, max_chars)

                if len(lines) > max_lines or duration > max_duration:
                    if len(words) > 1:
                        words = words[:-1]
                        end_time = chunks[i - 1]["timestamp"][1]
                    break

                i += 1

                if word and word[-1] in ".!?,;:":
                    end_time = chunks[i - 1]["timestamp"][1]
                    break

            if words:
                lines = self.__split_into_lines(words, max_chars)
                subtitle_text = "\n".join(lines[:max_lines])
                duration = end_time - start_time

                if duration < min_duration and i < len(chunks):
                    end_time = min(
                        start_time + min_duration,
                        chunks[min(i, len(chunks) - 1)]["timestamp"][1],
                    )

                subtitles.append((start_time, end_time, subtitle_text))

        # Set transcribed seconds to last timestamp in seconds
        self.__transcribed_seconds = subtitles[-1][1] if subtitles else 0.0
        self.__result = subtitles

        return subtitles

    def subtitles(self) -> str:
        """
        Generate SRT format subtitles from transcript chunks.
        """

        srt_output = []

        for idx, (start, end, text) in enumerate(self.__result, 1):
            srt_output.append(f"{idx}")
            srt_output.append(
                f"{self.__format_timestamp(start)} --> {self.__format_timestamp(end)}"
            )
            srt_output.append(text)
            srt_output.append("")

        return "\n".join(srt_output)

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

        diarization = self.__diarization_pipeline(
            self.__audio_path, num_speakers=int(self.__speakers)
        )

        aligned_segments = self.__align_speakers(self.__result, diarization)

        return {
            "segments": aligned_segments,
            "speaker_count": len(list(diarization.speaker_diarization.labels()))
            if diarization
            else 0,
        }

    def __align_speakers(self, transcription_chunks, diarization) -> list:
        """
        Align transcription chunks with speaker diarization results.
        """
        aligned_segments = []

        for chunk in transcription_chunks:
            chunk_start, chunk_end, chunk_text = chunk
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

        for segment, _, speaker in diarization.speaker_diarization.itertracks(
            yield_label=True
        ):
            if segment.start <= time_point <= segment.end:
                return speaker

        return "UNKNOWN"

    def __get_speakers_in_range(self, diarization, start_time, end_time) -> list:
        """
        Get a list of active speakers within a specific time range in the
        diarization.
        """
        active_speakers = set()

        for segment, _, speaker in diarization.speaker_diarization.itertracks(
            yield_label=True
        ):
            if not (segment.end < start_time or segment.start > end_time):
                active_speakers.add(speaker)

        return list(active_speakers)


if __name__ == "__main__":
    w = WhisperAudioTranscriber(
        logger=logging.getLogger(),
        audio_path="test2.wav",
        model_name="openai/whisper-base",
        language="en",
        speakers=2,
        hf_token=settings.HF_TOKEN,
    )

    w.transcribe()
    print(w.subtitles())
    print(w.diarization())
