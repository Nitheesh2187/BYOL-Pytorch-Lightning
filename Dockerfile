FROM python:3.9-slim

# Update pip
RUN pip install --upgrade pip

# Install pytorch-lightning
RUN pip install pytorch-lightning

# Copy your script into the container
COPY byol_light.py /app/byol_light.py

# Set the working directory
WORKDIR /app

# Run your script
CMD ["python", "byol_light.py"]