FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim

WORKDIR /api1

COPY . /api1

COPY /Users/judith/.cache/eotdl/creds.json ~/.cache/eotdl/creds.json

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api1:app", "--reload", "--host", "0.0.0.0"]