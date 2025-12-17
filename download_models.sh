#!/bin/sh

# Create models/ if needed
mkdir -p models

# Download Swedish and Norwegian models
wget -q --show-progress -O models/sv_${i}.bin https://huggingface.co/KBLab/kb-whisper-large/resolve/main/ggml-model-q5_0.bin
wget -q --show-progress -O models/no_${i}.bin https://huggingface.co/NbAiLab/nb-whisper-large/resolve/main/ggml-model-q5_0.bin

# Download large generic model
wget -q --show-progress -O models/whisper_large.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v2-q5_0.bin
