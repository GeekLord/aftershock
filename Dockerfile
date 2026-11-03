# Aftershock demo server. The precomputed mission needs no API key.
# Set MODEL_API_KEY to enable Run live and uploads; AFTERSHOCK_BUDGET_USD caps spend; set AFTERSHOCK_SPEND_FILE on persistent storage so the cap survives restarts.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080 AFTERSHOCK_BUDGET_USD=3 AFTERSHOCK_MAX_LIVE=3 AFTERSHOCK_IP_COOLDOWN_S=600
EXPOSE 8080
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
