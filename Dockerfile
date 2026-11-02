# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install Python dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --no-deps --no-cache-dir mediapipe==0.10.14
RUN pip install --no-cache-dir -r requirements.txt


RUN pip install python-multipart

# Expose port 8000
EXPOSE 8000

# Command to run your Python application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
