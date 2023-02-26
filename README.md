
![modelrocket logo](https://i.imgur.com/GsfsKTU.png)

Ben Guan, Vijay Shah, Jacob Stolker, Ryan Ziegler
# Overview
ModelRocket was designed from the ground up to be horizontally scalable, and is divided into n core services:
1. `Coordinator`: responsible for autoscaling and processing updates to authentication. Written in Python, interfaces with Redis.
2. `Model Manager`: responsible for processing new model uploads and getting them ready for deployment. Written in NodeJS, interfaces with the Coordinator and Redis.
3. `Inference Gateway`: responsible for routing inference requests for a model to an appropriate upstream server. Written in Python, interfaces with Redis.
4. `Inference Host`: custom software running on servers to manage active models and provide fast inference. Written in Python, interfaces with Redis.
5. `Python Client`: allows the user to upload their model to our service with a single line of code. Detects model type and converts it to ONNX, then makes the necessary post requests to store the model in MongoDB and S3. 
6. `Web UI`: allows users to manage their models and update API keys. Written in React.
# Client Setup
* Pushing model to our service
    * usage: modelrocket.upload(model, model_name, input_type, input_shape) 
        * This will output an api token and a model uuid.
* Making an inference request
    * usage: make a post request to the following link http://127.0.0.1:5000/inference with the following JSON body: 
    <pre>
    {
        "token": api_token, 
        "model": model_uuid, 
        "inputs": {"modelInput": your_input_to_the_model}
    }
    </pre>
    * This will output a JSON object with the results of your inference. 
* Website
    * Gives demonstration of how to write code for our service 
# Server Setup (required only if you want to host)
1. Install necessary packages for each service
    * `Coordinator`: 
        * run `cd coordinator`
        * run `pipenv install`
        * run `cd ..`
    * `Model Manager`: 
        * `cd model-manager`
        * Add a `.env` with the following variables:
            <pre>
            MONGODB_URL=<your_mongodb_url>
            ACCESS_KEY_ID=<your_s3_user>
            SECRET_ACCESS_ID=<your_s3_secret>
            </pre>
        * run `npm install`
    * `Inference Gateway`: 
        * run `cd coordinator`
        * run `pipenv install`
        * run `cd ..`
    * `Inference Host`: 
        * run `cd coordinator`
        * run `pipenv install`
        * run `cd ..`
2. Run `docker-compose up --detach`
3. Run the following commands to set up environment variables: 
    * `AWS_ACCESS_KEY_ID=<your_access_key_id> AWS_SECRET_ACCESS_KEY=<your_secret_access_key></your_secret_access_key> S3_ENDPOINT=http://localhost:9000/ PORT=5002 python host.py`
    * `REDIS_HOST=localhost REDIS_PORT=6379 REDIS_PASSWORD=<your_password> REDIS_DB=0 python -m flask --app gateway run`
    * `REDIS_HOST=localhost REDIS_PORT=6379 REDIS_PASSWORD=<your_password> REDIS_DB=0 python assigner.py`
    
4. S3 setup: 
    * Go to `localhost:9000`
    * Sign in with your access key-id and secret-key. 
    * Create a new bucket called `models`

