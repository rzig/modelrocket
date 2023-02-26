![modelrocket logo](https://i.imgur.com/GsfsKTU.png)

Ben Guan, Vijay Shah, Jacob Stolker, Ryan Ziegler
## Inspiration

ML models are becoming increasingly easy to create, but model deployment hasn't caught up yet. It can take hours for an ML developer to get their model deployed into an internet-accessible API, and forget making it scalable. We realized that this problem is widespread, and that we could do something about it.

<!-- A widespread problem throughout computer science is deploying ML models to applications. After all, one would expect that after training their model they would be done. However, figuring out how to run a model on different devices and how to run that model efficiently on the internet is an entirely different and highly complex task. We realized that an elegant solution would be to create a third party application that stored models on the cloud and ran them for the user. This would cut out the work of users deploying their model to a website, allowing them to focus on simply creating the best model they can.  -->

## What it does

With one line of Python, anyone can deploy their ML model to a scalable, internet-accessible inference API powered by ModelRocket. ModelRocket takes care of scaling the number of servers running the model, updating the model when you retrain it, and making sure the API is secure, enabling you to focus on building an amazing app-not on deploying it.

![img](https://i.imgur.com/vRWWHUD.png)

![img](https://i.imgur.com/V6q1K7p.png)


## How we built it


### Architecture overview
ModelRocket was designed from the ground up to be horizontally scalable, and is divided into 4 core backend services:
1. `Coordinator`: responsible for autoscaling and processing updates to authentication. Written in Python, interfaces with Redis.
2. `Model Manager`: responsible for processing new model uploads and getting them ready for deployment. Written in NodeJS, interfaces with the Coordinator and Redis.
3. `Inference Gateway`: responsible for routing inference requests for a model to an appropriate upstream server. Written in Python, interfaces with Redis.
4. `Inference Host`: custom software running on servers to manage active models and provide fast inference. Written in Python, interfaces with Redis.

ModelRocket also has two user-facing services:
1. `Web UI`: allows users to manage their models and update API keys. Written in React.
2. `Python Client`: consists of one function for users to upload their model to our service.
### Load balancing
A typical way to distribute tenants to servers in a distributed system is to use *sharding*, where the available servers are put into fixed groups of size *n*. We realized that this would not be feasible for our project, because we need to dynamically adjust the number of servers a tenant (model) is assigned to in response to load; and, due to variable model sizes, one large model could negatively impact the performance of the other models on its shard.

Research led us to discover a technique introduced by AWS called "Shuffle Sharding," in which each tenant is assigned a unique virtual shard; as a result, the probability of adverse impacts to users is reduced exponentially since as the number of servers increases because each shard is a random subset of the available servers. While AWS assigned tenants to fixed shards, we extended their ideas to dynamically grow and shrink each models' shards in response to model load.

The `Coordinator` service is the brain of ModelRocket and is responsible for listening to load and updating shuffle shards in real time; in particular, it keeps a rolling estimate of the requests per second in the last ten seconds, and uses this to update the shard assignments in Redis.

A Python gateway is responsible for routing inference requests to hosts, and reads Redis for each request it recieves to find a host. Thus the updates from the Coordinator are applied in realtime. Since Redis is an in-memory datastore, the overhead of these checks is only 1-2 milliseconds.

![img](https://i.imgur.com/Y4We6kt.png)

ModelRocket is a great tool for prototyping, but this means that it's likely the number of models we have will far exceed the number of servers we have; however most of these models will typically be used in short bursts of testing. We intentionally oversubscribe models on each host by using an LRU cache to evict models from a host when they become stale and we need to make room for a new model.

### Model Storage and Updates
We designed our model storage system with ease of updates in mind, since updates are a frequent use case for our system, especially in the prototyping phase. When a model is loaded onto a server for inference, the model's file is downloaded from an S3 bucket. As a result, the procedure for updating a model is as simple as evicting the model from any servers on which it is currently loaded; subsequent requests will automatically pull the new file.

Since we chose to be framework agnostic, this meant we'd need a standardized model format. We settled on ONNX because it's supported by all major machine learning libraries such as Tensorflow and PyTorch. There's also an existing ONNX runtime in Python and C++, so using ONNX makes it easy for us to run inference.

## Challenges we ran into

One of the major challenges we ran into was integrating all of the different components together that we articulated above. From just the initial Python program, information about the user and the model needed to be stored in MongoDB as well as an Amazon S3 Bucket. From these storage components, the `Coordinator` service could then add and remove inference hosts, as well as expand or shrink model shards depending on usage information from Redis. With so many moving parts that we each worked on separately, it was difficult to incorporate and wire everything back together.

Similarly, it was very difficult to run and test our service in the development process. We had to ensure everything was working on one computer after combining all of the components, while error checking various cases. The components had to work together and run in a very specifc order. 

## Accomplishments that we're proud of

* Getting a working front-end with digit recognition.
    * immediately functional with existing ML libraries & models
* Building the app in such a scalable way.
* Making a useful product that solves a real world problem. 
## What we learned

* We learned that all the common machine learning libraries can export to ONNX formatting. So, we learned how to convert machine learning models from various libraries to an ONNX format and how to run inference with ONNX models. 
* We learned efficient ways to handle multiple component integration including docker compose. 
* We gained experience using new tools and frameworks together, such as MongoDB, Express, Node, Amazon S3, and Flask, among many others across the frontend and backend.


## What's next for ModelRocket

* Scaling ModelRocket for more users and models
* Introducing payment tiers / plans to the service
* More error checking on storing user and model information, including max model sizes
* Deploying the server online to Azure

![website page](https://i.imgur.com/FunBxfK.png)
