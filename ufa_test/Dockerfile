FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]