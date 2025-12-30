import logging
import os
import shutil
import torch

from pyannote.audio import Pipeline, core
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional
from utils.settings import get_settings
from utils.subtitles import chunks_to_subs

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
        model_name: Optional[str] = "KBLab/kb-whisper-large",
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
            chunk_length_s=30,
            generate_kwargs={"task": "transcribe", "language": self.__language},
        )

        self.__process_transcription(result.get("chunks", []))

        return self.__transcribed_seconds

    def subtitles(self) -> str:
        """
        Generate SRT format subtitles from transcript chunks.
        """

        return chunks_to_subs(self.__result)

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

        # Rename speakers to Speaker_1, Speaker_2, ...
        speakers = {}

        for segment in aligned_segments:
            speaker_label = segment["speaker"]

            if speaker_label not in speakers:
                speakers[speaker_label] = f"SPEAKER_{len(speakers) + 1}"

            segment["speaker"] = speakers[speaker_label]

            print(
                f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['speaker']} - {segment['text']}"
            )

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
            chunk_text = chunk_text.replace("\n", "").strip()
            dominant_speaker = self.__get_speaker(diarization, chunk_middle)
            active_speakers = self.__get_speakers_in_range(
                diarization, chunk_start, chunk_end
            )

            aligned_segments.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": chunk_text,
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
        audio_path="test.wav",
        model_name="kblab/kb-whisper-large",
        language="sv",
        speakers=0,
        hf_token=settings.HF_TOKEN,
    )

    w.transcribe()
    print(w.subtitles())
    # print(w.diarization())
