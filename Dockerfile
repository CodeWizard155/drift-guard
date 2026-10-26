FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/

# Default: run tests, then the full monitoring pipeline and charts
CMD ["sh", "-c", "python -m unittest && python -m src.pipeline && python -m src.visualize"]
