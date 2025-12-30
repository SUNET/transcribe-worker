import re

from typing import List, Tuple

MAX_CHARS = 42
MAX_LINES = 2
MIN_DURATION = 1.0
MAX_DURATION = 7.0
MIN_GAP = 0.2


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    """

    if not seconds:
        return "00:00:00,000"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def is_sentence_break(text: str) -> bool:
    return bool(re.search(r"[.!?]$", text.strip()))


def split_lines(text: str) -> str:
    """
    Split text into max two lines of <= 42 chars.
    """
    words = text.split()
    lines = []
    current = ""
    remaining_text = ""
    i = 0

    while i < len(words):
        w = words[i]
        if len(current) + len(w) + 1 <= MAX_CHARS:
            current = f"{current} {w}".strip()
            i += 1
        else:
            lines.append(current)
            current = w
            i += 1
            if len(lines) == MAX_LINES:
                remaining_text = " ".join(words[i:])
                break

    if current and len(lines) < MAX_LINES:
        lines.append(current)
    elif current:
        remaining_text = " ".join([current] + words[i:])

    return "\n".join(lines), remaining_text


def chunks_to_subs(chunks: list) -> str:
    """
    Convert word-level chunks into SRT subtitles following BBC guidelines.
    """

    subs: List[Tuple[float, float, str]] = []

    buffer_words = []
    start_time = None
    last_end = 0.0
    srt_output = []

    for i, chunk in enumerate(chunks):
        text = chunk["text"].strip()
        start, end = chunk["timestamp"]
        start_time = None
        end_time = None

        if start_time is None:
            start_time = start

        buffer_words.append(text)
        current_text = " ".join(buffer_words)
        duration = end - start_time

        too_long = duration >= MAX_DURATION
        natural_break = is_sentence_break(text)

        gap_after = (
            i + 1 < len(chunks) and chunks[i + 1]["timestamp"][0] - end >= MIN_GAP
        )

        if (
            len(current_text) >= MAX_CHARS * MAX_LINES
            or too_long
            or (natural_break and duration >= MIN_DURATION)
            or gap_after
        ):
            end_time = max(end, start_time + MIN_DURATION)
            end_time = min(end_time, start_time + MAX_DURATION)

            if start_time - last_end < MIN_GAP and subs:
                start_time = last_end + MIN_GAP

            split_text, remaining_text = split_lines(current_text)
            subs.append((start_time, end_time, split_text))

            last_end = end_time
            buffer_words = remaining_text.split()
            start_time = None

    if buffer_words:
        text = split_lines(" ".join(buffer_words))
        end_time = chunks[-1]["timestamp"][1]
        if end_time - start_time < MIN_DURATION:
            end_time = start_time + MIN_DURATION
        subs.append((start_time, end_time, text))

    for index, (start, end, text) in enumerate(subs, start=1):
        srt_output.append(f"{index}")
        srt_output.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_output.append(f"{text}\n")

    return "\n".join(srt_output)
