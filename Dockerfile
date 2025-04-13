FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code
COPY . .

# Make the entrypoint script executable
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Set the entrypoint to the bash script
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]