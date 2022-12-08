FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04
WORKDIR /application

# Install and upgrade things
RUN apt-get update && apt-get upgrade
RUN pip install --upgrade pip

# Collect pip requirements
COPY requirements.txt .

# Install pip requirements
RUN pip install -r requirements.txt

# Copy the code
COPY SFA3D-master/ .

# add non-root user
RUN useradd appuser && chown -R appuser /application
USER appuser

# train SFA3D model
CMD ["python", "sfa/train.py", "--gpu_idx", "0"]

