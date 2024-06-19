from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(1, 2.5)

    @task
    def post_image(self):
        with open("examples/samples/deep_globe.jpg", "rb") as f:
            self.client.post("/EuroSAT-RGB-Q2", files={"image": f})
