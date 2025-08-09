FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl build-essential cmake libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --break-system-packages
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

RUN pip install --no-cache-dir --force-reinstall \
    git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

COPY app.py /app/app.py

CMD ["python3", "app.py"]
