FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
#FROM python:3.8.5
WORKDIR /application

#needed for tzdata
ENV TZ=Europe/Brussels


# aditional opencv-python requirements
RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN sudo apt-get install ffmpeg libsm6 libxext6  -y

#hier moet de dataset in komen
VOLUME /Dataset
#Hier moet de file alles in wegschrijven; --> Snapshots elke 5 epochs en gegenereerd model
VOLUME /Snapshots


# Install and upgrade things
RUN pip install --upgrade pip

# Collect pip requirements
COPY requirements.txt .

# Install pip requirements
RUN pip install -r requirements.txt
RUN pip install protobuf==3.20.* #for some reason a problem...

# Copy the code
COPY SFA3D-master/ .

# add non-root user
#RUN useradd appuser && chown -R appuser /application
#USER appuser

# train SFA3D model
CMD ["/bin/bash"]
#CMD ["python", "sfa/train.py", "--gpu_idx", "0"]

