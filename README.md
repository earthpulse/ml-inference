# ML Inference Processor

This repository contains resources for creating production-grade ML inference processors. Models are expected to be hosted on the [EOTDL](https://www.eotdl.com/) as [Q2+ models](https://github.com/earthpulse/eotdl/blob/main/tutorials/notebooks/06_q2_model.ipynb) ([ONNX](https://onnx.ai/) models + [STAC](https://stacspec.org/en) metadata with the [MLM](https://github.com/crim-ca/mlm-extension) extension). The following features are included:

- [x] CPU/GPU inference
- [x] Docker 
- [x] Kubernetes
- [x] Auto-scaling
- [x] Load testing

Future features will include:

- [ ] Monitoring & Alerting
- [ ] Batch & Online processing
- [ ] Data drift detection
- [ ] Security & Safety

## Running the default processors

To run the default API with `Docker`, you can use the following command:

```bash
# cpu
docker run -p 8000:80 -e EOTDL_API_KEY=<eotdl_api_key> earthpulseit/ml-inference

# gpu
docker run --gpus all -p 8000:80 -e EOTDL_API_KEY=<eotdl_api_key> earthpulseit/ml-inference-gpu
```

> You can get your EOTDL API key for free by signing up at [EOTDL](https://www.eotdl.com/) and creating a new token in your profile.

You can also use the sample `k8s` manifests to deploy the API to a Kubernetes cluster.

```bash
kubectl apply -f k8s/deployment.yaml
```

## Building a new processor

This repository offers functionality for creating production-grade APIs to perform inference on ML models. 


### Developing locally

To develop the api, run 

```
# cpu support
docker-compose -f docker-compose.cpu.yaml up

# gpu support
docker-compose -f docker-compose.gpu.yaml up
```

You can try the api with the interactive documentation at `http://localhost:8000/docs`.

### Running in production

#### Create a docker image

Build the docker image:

```bash
# cpu
 docker build -t <username>/<image-name>:<tag> api

# gpu
 docker build -t <username>/<image-name>:<tag> -f api/Dockerfile.gpu api
```

> Use your dockerhub username and a tag for the image.

Push to docker hub:

```
docker push <username>/<image-name>:<tag>
```

> You will need to login to dockerhub with your credentials before pushing the image.

You can run the image with:

```bash
# cpu
docker run -p 8000:8000 <username>/<image-name>:<tag>

# gpu
docker run --gpus all -p 8000:8000 <username>/<image-name>:<tag>
```

### Run with kubernetes

#### MiniKube

Start minikube:

```bash
# cpu
minikube start

# gpu
minikube start --kvm-gpu
```

> add metrics server if you want to use autoscaling `minikube addons enable metrics-server`

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

#### Autoscaling

You can autoscale your API with the following command:

```bash
kubectl apply -f k8s/hpa.yaml
```

Modify the manifest to adjust to your needs.

You can test the autoscaling with a load test with `locust`.

#### GPU

Minikube gpu support is limited. Following https://minikube.sigs.k8s.io/docs/tutorials/nvidia/ does not seem to work.

- Deployment: install nvidia device plugin in k8s nodes and add `resources: limits: nvidia.com/gpu: 1` to the deployment manifest. No more GPUs than available on nodes can be used.
- Autoscaling: expose gpu usage as custom metric (prometheus + node exporter + prometheus adapter) and use it in the hpa manifest.