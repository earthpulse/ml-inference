# ML Inference Processor

## Developing

To develop the api, run 

```
docker-compose up
```

You can try the api with the interactive documentation at `http://localhost:8000/docs`.

## Running in production

### Create a docker image

Build the docker image:

```bash
 docker build -t <username>/<image-name>:<tag> .
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