FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home appuser
WORKDIR /home/appuser

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project and install python deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy project files
COPY . /home/appuser/AI_MICROSCOPE
WORKDIR /home/appuser/AI_MICROSCOPE

# Ensure records dir exists
RUN mkdir -p model/records exports

USER appuser

# Default entrypoint runs GUI; for headless servers use scripts provided
ENTRYPOINT ["python", "app/main_app.py"]
