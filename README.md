# ML Inference Processor

This repository contains resources for creating production-grade ML inference processors. Models are expected to be hosted on the [EOTDL](https://www.eotdl.com/) as [Q2+ models](https://github.com/earthpulse/eotdl/blob/main/tutorials/notebooks/06_q2_model.ipynb) ([ONNX](https://onnx.ai/) models + [STAC](https://stacspec.org/en) metadata with the [MLM](https://github.com/crim-ca/mlm-extension) extension). The following features are included:

- [x] CPU/GPU inference
- [x] Docker 
- [x] Kubernetes
- [x] Auto-scaling
- [x] Load testing
- [x] Batch & Online processing
- [x] Monitoring & Alerting

Future features will include:

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

### Batch processing

By default, requests to the API are processed sequentially. You can change this behavior by setting the `BATCH_SIZE` and `BATCH_TIMEOUT` environment variables. 

- `BATCH_SIZE`: Maximum number of requests to process in a single batch.
- `BATCH_TIMEOUT`: Maximum time (in seconds) to wait before processing an incomplete batch.

```bash
# cpu
docker run -p 8000:80 -e EOTDL_API_KEY=<eotdl_api_key> -e BATCH_SIZE=<batch_size> -e BATCH_TIMEOUT=<batch_timeout> earthpulseit/ml-inference

# gpu
docker run --gpus all -p 8000:80 -e EOTDL_API_KEY=<eotdl_api_key> -e BATCH_SIZE=<batch_size> -e BATCH_TIMEOUT=<batch_timeout> earthpulseit/ml-inference-gpu
```

> This setting is particularly useful if the number of concurrent requests exceeds the time it takes to run inference with a model on your hardware.

### Monitoring and alerting

In order to monitor the API, you can use Prometheus and Grafana. For this case we recommend using the `docker-compose.minitoring.yaml` file or the corresponding `k8s` deployments.

```bash
docker compose -f docker-compose.monitoring.yaml -f docker-compose.cpu.yaml up
```

In Grafana:
- Add Prometheus as a data source (URL: http://prometheus:9090)
- Import dashboard for FastAPI monitoring (you can start with dashboard ID 18739 from Grafana's dashboard marketplace)

This setup will give you:
- Basic metrics like request count, latency, and status codes
- System metrics like CPU and memory usage
- Custom metrics that you can add later
- Visualization and alerting capabilities through Grafana

You can further customize the monitoring by:
- Adding custom metrics in your FastAPI code
- Creating custom Grafana dashboards
- Setting up alerts in Grafana
- Adding more Prometheus exporters for system metrics

Some custom metrics included are:
- `model_counter`: Number of models requested
- `model_error_counter`: Number of errors in model inference
- `model_inference_duration`: Time spent processing inference requests
- `model_inference_batch_size`: Number of images in the batch
- `model_inference_timeout`: Number of inference requests that timed out

You can set alerts in Grafana for these metrics by going to `Alerting > Alert Rules` in the Grafana dashboard. Some examples are:
- `rate(model_inference_errors_total[1m]) > 10`: Alert if more than 10 errors in the last minute
- `histogram_quantile(0.95, rate(model_inference_duration_seconds[1m])) > 10`: Alert if more than 95% of the requests take more than 10 seconds to process
- You can add more alerts using `PromQL` queries.

> TODO: It may be interesting to create a custom dashboard with the included metrics and share it through Grafana's dashboard marketplace.

Alternatively, you can use `AlertManager` to send notifications via email, Slack, etc. (out of the scope of this repository).

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
minikube start
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

#### Monitoring

You can monitor the API with Prometheus and Grafana.

```bash
kubectl apply -f k8s/prometheus-config.yaml
kubectl apply -f k8s/monitoring.yaml
```

Port forward to access Prometheus and Grafana:

```bash
kubectl port-forward service/prometheus-service 9090:9090
kubectl port-forward service/grafana-service 3000:3000
```

Connect Prometheus to Grafana with `http://prometheus-service:9090` as the Prometheus instance.

#### GPU

Minikube gpu support is limited. Following https://minikube.sigs.k8s.io/docs/tutorials/nvidia/ does not seem to work.

- Deployment: install nvidia device plugin in k8s nodes and add `resources: limits: nvidia.com/gpu: 1` to the deployment manifest. No more GPUs than available on nodes can be used.
- Autoscaling: expose gpu usage as custom metric (prometheus + node exporter + prometheus adapter) and use it in the hpa manifest.

#### Running in the cloud

To run the api in a cloud kubernetes cluster, you should follow the same steps as for minikube taking the following into account:

- Change the service type to `LoadBalancer` or `ClusterIP` in the deployment manifests service section.
- Use a cloud provider that supports gpu nodes (if you want to use the gpu version).
- Use `ingress.yaml` to expose the api to the internet instead of port forwarding.