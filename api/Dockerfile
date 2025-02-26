FROM python:3.8-slim

WORKDIR /app

RUN apt-get update -y && apt-get install -y curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

RUN uv venv

# need to install gdal to make rasterio work in macos

# RUN apt-get update -y && apt-get install -y g++ gdal-bin libgdal-dev

# ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
# ENV C_INCLUDE_PATH=/usr/include/gdal

# RUN pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`

COPY requirements.txt requirements.txt
RUN uv pip install -r requirements.txt

COPY requirements.cpu.txt requirements.cpu.txt
RUN uv pip install -r requirements.cpu.txt

COPY src /app/api/src
COPY main.py /app/api/main.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
