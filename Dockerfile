FROM nvcr.io/nvidia/pytorch:24.01-py3

COPY . /workspace

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Install lerobot without its heavy dependencies (gymnasium, accelerate,
# rerun-sdk, torchcodec, etc.) to avoid version conflicts.  The dataset
# classes used by X-MOBILITY only need packages already in requirements.txt
# plus the few extras added there (datasets, huggingface-hub, av).
RUN pip install lerobot==0.3.3 --no-deps
