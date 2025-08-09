FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git curl build-essential cmake ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libcublas-dev libcufft-dev libcurand-dev libcusolver-dev libcusparse-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv .env
ENV VIRTUAL_ENV=/app/.env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

ENV MAX_JOBS=8
RUN pip install --no-cache-dir --force-reinstall \
    git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

COPY app.py /app/app.py

CMD ["python3", "app.py"]
