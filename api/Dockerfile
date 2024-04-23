FROM continuumio/miniconda3:22.11.1

WORKDIR /app

ADD . /app

# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -U rasterio

RUN conda config --add channels conda-forge && conda update -y conda \
    && conda install -y geopandas
RUN conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive
RUN conda install -y -c conda-forge --file requirements.txt
RUN pip install eotdl

RUN pip install eotdl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
