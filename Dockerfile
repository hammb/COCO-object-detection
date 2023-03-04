FROM python:3.10-slim

# Install pip requirements
COPY files/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy application files
COPY files/flask_server.py /app/flask_server.py

WORKDIR /app

RUN apt-get update && apt-get install -y git \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install git-lfs \
    && git lfs install \
    && git clone https://huggingface.co/facebook/detr-resnet-50

# Run the application
CMD ["python", "flask_server.py"]