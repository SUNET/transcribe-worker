#!/bin/sh

# Create models/ if needed
mkdir -p models

# Download the KB models
for i in tiny base large; do
	wget -q --show-progress -O models/sv_${i}.bin https://huggingface.co/KBLab/kb-whisper-${i}/resolve/main/ggml-model-q5_0.bin
done

# Download other models
for i in tiny base; do
	wget -q --show-progress -O models/whisper_${i}.bin https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-${i}-q5_1.bin
done

wget -q --show-progress -O models/whisper_large.bin https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v2-q5_0.bin
