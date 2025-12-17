FROM ghcr.io/astral-sh/uv:debian

# Install dependencies
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	ffmpeg \
	cmake \
	gcc \
	nvidia-cuda-gdb \
	nvidia-cuda-toolkit \
	git && \
	rm -rf /var/lib/apt/lists/*

# Build whisper.cpp
RUN git clone https://github.com/ggml-org/whisper.cpp.git
WORKDIR /whisper.cpp

RUN git checkout v1.8.2

RUN cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86"
RUN cmake --build build -j --config Release

# Copy code
WORKDIR /app
COPY . .

# Run worker
CMD ["uv", "run", "main.py", "--foreground", "--debug", "--no-healthcheck"]
