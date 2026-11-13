FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY models/best_valve_model.joblib ./models/best_valve_model.joblib
COPY data/features_all_cycles.csv data/profile.txt ./data/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT','8000'))"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 app.app:app"]
