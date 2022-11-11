ARG VARIANT=3.10
# FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}
# For GPU support use the following instead of the above:
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive
    
# Add PIP
RUN apt-get update && apt-get -y install python3-pip && pip install pip==22.3
# Fix GPU related problems by installing a torch version that comes bundled with a newer CUDA version
RUN pip3 --no-cache-dir install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

COPY ../requirements.txt /code/requirements.txt
WORKDIR /code
RUN pip install -r requirements.txt
COPY . /code


