# Containerised local fallback for the demo.
#   docker build -t anon-feedback .
#   docker run -p 8501:8501 --env-file .env anon-feedback
# then open http://localhost:8501
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
