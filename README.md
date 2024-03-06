# ml-inference
ML Inference Processor

A docker image has been created and runned using the Dockerfile:

```bash
 docker build -t ml-inference .
```

```bash
 docker run --name ml-inference -p 8000:8000 ml-inference
```

Or altrenatively, using the docker compose configuration file:

```bash
docker-compose up
```

The image has been pushed to dockerhub with this commands:

```bash
docker tag ml-inference judithep/ml-inference:v1.0
```

```bash
docker push judithep/ml-inference:v1.0
```