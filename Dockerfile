FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11-slim

WORKDIR /api1

COPY . /api1

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0"]