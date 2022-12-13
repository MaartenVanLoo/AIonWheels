#Docker run command:
#docker run -d --name=AIonWheels --mount type=bind,source=<DATASET_DIR>,target=/application/dataset --mount type=bind,source=<CHECKPOINTS_DIR>,target=/application/checkpoints onzeImage

FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
#FROM python:3.8.5
WORKDIR /application

#needed for tzdata
ENV TZ=Europe/Brussels

# aditional opencv-python requirements
RUN sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN sudo apt-get install ffmpeg libsm6 libxext6  -y

#Add dataset in this directory
#VOLUME /Dataset
#Write results in this dir
#VOLUME /Snapshots


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

