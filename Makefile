run:
	docker-compose up

stop:
	docker-compose down

build:
	docker build -t ${u}/ml-inference:${v} api/.

push:
	docker push ${u}/ml-inference:${v}

minikube:
	minikube start

deploy:
	kubectl apply -f k8s/deployment.yaml

service:
	# minikube service ml-inference-service --url 
	kubectl port-forward service/ml-inference-service 8000:80 