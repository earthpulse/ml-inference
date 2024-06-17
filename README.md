# ML Inference Processor

This repository contains resources for creating production-grade ML inference processors. Models are expected to be hosted on the [EOTDL](https://www.eotdl.com/) as [Q2+ models](https://github.com/earthpulse/eotdl/blob/main/tutorials/notebooks/05_q2_model.ipynb) ([ONNX](https://onnx.ai/) models + [STAC](https://stacspec.org/en) metadata with the [MLM](https://github.com/crim-ca/mlm-extension) extension). The following features are included:

- [x] CPU/GPU inference
- [x] Docker 
- [x] Kubernetes

Future features will include:

- [] Auto-scaling
- [] Monitoring & Alerting
- [] Batch & Online processing
- [] Data drift detection
- [] Security & Safety
- [] Load testing

## Running the default processors

To run the default API with `Docker`, you can use the following command:

```bash
docker run -p 8000:80 -e EOTDL_API_KEY=<eotdl_api_key> earthpulse/ml-inference
```

> You can get your EOTDL API key for free by signing up at [EOTDL](https://www.eotdl.com/) and creating a new token in your profile.

Other images include GPU support and other features

```bash
docker run -p 8000:80 -e EOTDL_API_KEY=<eotdl_api_key> earthpulse/ml-inference-gpu
```

You can also use the sample `k8s` manifests to deploy the API to a Kubernetes cluster.

```bash
kubectl apply -f k8s/deployment.yaml
```

## Building a new processor

This repository offers functionality for creating production-grade APIs to perform inference on ML models. 


### Developing locally

To develop the api, run 

```
docker-compose up
```

You can try the api with the interactive documentation at `http://localhost:8000/docs`.

### Running in production

#### Create a docker image

Build the docker image:

```bash
 docker build -t <username>/<image-name>:<tag> api
```

> Use your dockerhub username and a tag for the image.

Push to docker hub:

```
docker push <username>/<image-name>:<tag>
```

> You will need to login to dockerhub with your credentials before pushing the image.

You can run the image with:

```bash
 docker run -p 8000:80 <username>/<image-name>:<tag>
```

### Run with kubernetes

#### MiniKube

Start minikube:

```bash
minikube start
```

Create secrets

```
kubectl create configmap ml-inference-config --from-env-file=.env
```

Deploy api to cluster:

```bash
kubectl apply -f k8s/deployment.yaml
```

> Change the image name in the deployment file to the one you created.

Port forward to access the api:

```bash
kubectl port-forward service/ml-inference-service 8000:80 
```

Get api logs

```bash
kubectl logs -f deployment/ml-inference
```