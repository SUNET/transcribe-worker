import torch

from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq
from transformers import AutoProcessor
from transformers import pipeline
from typing import Optional


class WhisperAudioTranscriber:
    def __init__(
        self,
        audio_path: str,
        model_name: Optional[str] = "KBLab/kb-whisper-base",
        language: Optional[str] = "sv",
        hf_token: Optional[str] = None,
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

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.__model_name,
            torch_dtype=self.__torch_dtype,
            low_cpu_mem_usage=False,
            use_safetensors=True,
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
            generate_kwargs={
                "max_new_tokens": 400,
                "language": self.__language,
                "task": "transcribe",
            },
            chunk_length_s=30,
            language=self.__language,
        )

        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.__hf_token
        )
        self.diarization_pipeline.to(torch.device(self.__device))

    def transcribe(self) -> list:
        """
        Transcribe the audio file using the Whisper model.
        """
        self.__result = self.pipe(
            self.__audio_path,
            chunk_length_s=30,
            generate_kwargs={"task": "transcribe", "language": self.__language},
        )

        return self.__result

    def diarization(self) -> dict:
        """
        Perform speaker diarization on the transcribed audio.
        """
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
                "full_transcription": self.__result["text"],
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
            print("Using CUDA")
            return "cuda:0", torch.float16
        elif torch.backends.mps.is_available():
            print("Using MPS")
            return "mps", torch.float16
        else:
            print("Using CPU")
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
