FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

CMD ["python3", "app.py"]