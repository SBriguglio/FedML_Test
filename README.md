# FL Platform Test: FedML

This repository is the home of my evaluation of the Flower federated learning platform; the aim of which is to better understand 
the extensibility and ease of use of the platform from the researcher's perspective. Specifically, I wish to address the
following questions:
1. How easy is it to implement different ML models in a FL setting using this platform?
2. Is there a large selection of ML models natively supported by the platform?
3. How easy is it to use different FL architectures (i.e. HFL, VFL, etc.)?
4. Can a custom ML model be used with this platform? If so, what process must be followed and how time-consuming is it?
5. What security features are supported by Flower? Can they be easily modified or extended?

The complete Flower documentation can be found here: [https://flower.dev/docs/index.html]()

## Environment
- Ubuntu 20.04
- Python 3.10.0
- Flower 0.17.0

Instillation Scripts:

*Additional dependencies for some tests (i.e. AutoML, sklearn, etc.)*
## Evaluation
Ongoing.
### Quickstart Tutorials
- generally easy to use for simple setups
- intuitive client and server based code
- easy to connect clients and server on the same network, just need the server's IP address (so 
it's a good idea to make sure it is static)
- unsure about client's ability to talk to server if it is not in the local network
- not immediately clear how to extend client-server communication protocols or to implement secure
protocols
- work with a variety of common ML libraries (TF, PyTorch, sklearn, etc.)
- not immediately clear what the util.py file does or how to implement something yourself
- if a new version of Flower needs requires an updated depency, but a package used in the code does
not yet support the new version of the dependency, then, practically speaking, Flower cannot be 
updated either

### Evaluation and Strategies
#### Evaluation
- two main approaches to model evaluation:
  1. server-side (centralized)
  2. federated
- ALL built in strategies support centralized evaluation via an evaluation function
  - the evaluation function is any function that can take the current global model parameters as
input and return evaluation results
  - the ```Strategy``` abstraction provides the ```evaluate``` method that can directly be used to
evaluate current global model parameters. The server-side implementation calls ```evaluate``` after
parameter aggregation and before federated evaluation
- Federated evaluation can be implemented by configuring the ```Client.evaluate``` method client-side
- Model parameters can also be evaluated during training by using the ```Client.fit``` method to
return arbitrary evaluation results as a dictionary
- [read more about evaluation here](href=https://flower.dev/docs/evaluation.html)
#### Strategies
- the ```Strategy``` abstraction allows full customization of the learning process
- various built-in strategies are already provided in the core framework
- three ways to customize the way Flower orchestrates the learning process on the server-side
    1. via an existing strategy (i.e. ```FedAvg```)
    2. customizing an existing strategy with callback functions
    3. implementing a novel strategy
- strategies are passed to the ```flwr.server.start_server``` function
- it is recommended to adjust ```start_server``` parameters during instantiation
- callback functions allow strategies to call user-provided code during execution
- the server can pass new configuration values to the clients each round by passing a function to
```on_fit_config_fn``` which will then by called by the strategy and must return a dictionary
configuration key that is sent to the client
  - the dictionary returned contains arbitrary configuration values ```client.fit``` and ```client.evaluate```
functions during each round of federate learning
- ```on_fit_config_fn``` can be used to send a new learning rate of each client which receives the
- dictionary returned by ```on_fit_config_fn``` in its own ```client.fit()``` function
- server-side evaluation can be enabled by passing an evaluation function to ```eval_fn```
- [read more about strategies here](https://flower.dev/docs/strategies.html)
#### Implementing Strategies
- online documentation incomplete [view](https://flower.dev/docs/implementing-strategies.html)
- ```Strategy``` abstraction enables fully customizable strategies
- a strategy is the federated learning algorithm that runs on the server, it decides how to sample 
clients, how to configure clients for training, how to aggregate updates and how to evaluate models
- ```Strategy``` is also the basis for all built in strategies and are derived from the 
```flwr.server.strategy.Strategy``` abstract base class
- creating a new strategy means implementing a new class derived from the abstract base class ```Strategy```
##### The ```Strategy``` base class:
```python
class Strategy(ABC):

    @abstractmethod
    def configure_fit(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

    @abstractmethod
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:

    @abstractmethod
    def configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

    @abstractmethod
    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:

    @abstractmethod
    def evaluate(self, weights: Weights) -> Optional[Tuple[float, float]]:
```
##### Example Implementation
```python
class SotaStrategy(Strategy):

    def configure_fit(self, rnd, weights, client_manager):
        # Your implementation here

    def aggregate_fit(self, rnd, results, failures):
        # Your implementation here

    def configure_evaluate(self, rnd, weights, client_manager):
        # Your implementation here

    def aggregate_evaluate(self, rnd, results, failures):
        # Your implementation here

    def evaluate(self, weights):
        # Your implementation here
```
### Saving Progress
- no automatic persistence or saving of model updates
- no persistence or saving of model updates or evaluation results server-side
- can be implemented by customizing ```Strategy```
- customizing ```Strategy``` can also be used to aggregate custom evaluation results coming from
individual clients
- clients can return custom metrics to the server by returning a dictionary
### SSL Encryption
- complete code example can be found [here](https://github.com/adap/flower/tree/main/examples/advanced_tensorflow)
to supplement this guide.
- can generate self-signed certificates using a provided script to start an SSL-enabled server
and have client establish a secure connection with it
#### Certificates
- requires in-depth review which will be conducted in the future
### Suggested Example: Centralized to Federated with PyTorch
- A fair summary example of this guide
- note: PyTorch will require Python < 3.10.0
- Available [here](https://flower.dev/docs/example-pytorch-from-centralized-to-federated.html)
## The Bottom Line
Flower is a fairly easy-to-use and extensible federated learning platform. It can be very easily installed
using pip and has good extensibility without the need to modify sourcecode. In my trials, I have
been able to quickly implement and train with as many as 10 clients across 3 virtuals machines and
as many as 60 clients and one server on a single machine in suboptimal conditions. It is quite
easy to implement various ML models using various different packages like sklearn, TensorFlow and 
PyTorch. Unfortunately, there does not seem to be any easy way to implement other federated learning
architectures. From the documentation and inspection of source code, it is not known
how other federated learning architectures are best implemented and how to better secure client-server
communications. It is not known what security measures are supported by Flower other than SSL
encryption.Hello

