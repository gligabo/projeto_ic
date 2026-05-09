FROM nvidia/cuda:12.6.1-devel-ubuntu24.04

WORKDIR /app

RUN apt update && \
    apt install -y python3 && \
    apt install -y python3-pip && \
    apt install -y python3-venv

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install jupyter && \
    pip install torch && \
    pip install matplotlib && \
    pip install scikit-image
COPY . /app
