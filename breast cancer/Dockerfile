from python:3.7-slim

# RUN command is needed to run any linux commmand in Dockerfile
# Create directory
RUN mkdir -p /app

# Copy the file from local machine working directory to docker directory
COPY . /app

# Change directory
WORKDIR /app

# Install dependencies
RUN pip3 --no-cache-dir install -r /app/requirement.txt

EXPOSE 5000

CMD ["python3", "/app/Flask_request.py"]
