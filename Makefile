run-cpu:
	docker-compose -f docker-compose.cpu.yaml up

run-gpu:
	docker-compose -f docker-compose.gpu.yaml up

build-cpu:
	sed -i 's/__version__ = '.*'/__version__ = "${v}"/' api/main.py
	docker build -t earthpulseit/ml-inference api/.

build-gpu:
	sed -i 's/__version__ = '.*'/__version__ = "${v}"/' api/main.py
	docker build -t earthpulseit/ml-inference-gpu -f api/Dockerfile.gpu api/.

push-cpu:
	docker push earthpulseit/ml-inference

push-gpu:
	docker push earthpulseit/ml-inference-gpu

minikube:
	minikube start

deploy:
	kubectl apply -f k8s/deployment.yaml

service:
	# minikube service ml-inference-service --url 
	kubectl port-forward service/ml-inference-service 8000:80 