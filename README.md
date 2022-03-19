# FL Platform Test: FedML

This repository is the home of my evaluation of the FedML federated learning platform; the aim of which is to better understand 
the extensibility and ease of use of the platform from the researcher's perspective. Specifically, I wish to address the
following questions:
1. How easy is it to implement different ML models in a FL setting using this platform?
2. Is there a large selection of ML models natively supported by the platform?
3. How easy is it to use different FL architectures (i.e. HFL, VFL, etc.)?
4. Can a custom ML model be used with this platform? If so, what process must be followed and how time-consuming is it?
5. What security features are supported by Flower? Can they be easily modified or extended?

The complete Flower documentation can be found here: [https://fedml.ai/](), [https://github.com/FedML-AI/FedML]() and [https://doc.fedml.ai/#/intro]()

## Environment
- Ubuntu 20.04
- Python 3.7.0

### Installation Instructions
FedML is deployable to three computing paradigms. 
1. Standalone Simulation
2. Distributed computing (i.e. ComputeCanada)
3. Mobile On-Device Training (Mobile and IoT)
#### Standalone Simulation
1. Clone GitHub repo for FedML: [https://github.com/FedML-AI/FedML.git](https://github.com/FedML-AI/FedML.git)
2. Run ```Ci-install.sh```
   1. Review the shell script first! You may need to modify the shell script to work
with your current environment (i.e. if Miniconda is already installed)
3. From here you can test the platform using their provided experiments in the ..\FedML\fedml_experiments\standalone 
folder or use the platform as desired.

Detailed instructions can be found here: https://doc.fedml.ai/#/installation
## Evaluation
Ongoing.
### Worker-Oriented Programming Interface
- helpful introduction video [here](https://www.youtube.com/watch?v=93SETZGZMyI)
- well organized tutorial experiments easily ran with convenient shell script
- unfortunately not very well documented compared to other platforms
- quickstart tutorials found throughout the FedML folder (everything in one place though not as easy to navigate)
- good place to start reading [here](https://arxiv.org/pdf/2007.13518.pdf) (the source for
most of the information in this document)
- very easy to scale up from standalone simulation to distributed computed and mobile deployments

### Library Architecture
<img src="https://github.com/SBriguglio/FedML_Test/blob/master/image_resources/fedml_architecture.jpg?raw=true"/></br>
[image source: https://github.com/FedML-AI/FedML/blob/master/docs/image/fedml.jpg](https://github.com/FedML-AI/FedML)
- The platform contains a number of packages for various use cases
- Has two key components:
    1. FedML-API which acts as the library's high-level API
    2. FedML-core acting as its Low-level API

#### FedML-core
- contains two separate modules. One handles distributed communication and the other handles model training
- communication backend is based on message passing interface (MPI)
- ```TopologyManager``` within the distributed communication module, supports a variety of network topologies
- security/privacy/related functions are also supported
- the model training module is build upon PyTorch
- users can implement workers(trainers) and coordinators according to their needs

#### FedML-API
- built upon FedML-core
- new algorithms in the distributed version can be easily implemented by adopting the client-oriented 
programming interface which enables flexible distributed computing
  - essential for scenarios in which large DNNs cannot be handles by standalone simulation due to GPU 
memory limitations and time constraints
  - also useful for conventional in-cluster large-scale distributed training
- separates the implementations of models, datasets, and algorithms
- enables code reuse and fair comparison, avoiding statistical or system-level gaps between 
algorithms led by non-trivial implementation differences
- can develop more models and submit more realistic datasets without the need to understand the 
details of different distributed optimization algorithms

#### Programming Interface
- provides simple user experience to allow users to build distributed training applications by only
focusing on algorithmic implementations while ignoring the low-level communication backend details

##### Worker/Client-Oriented Programming
- used to program worker behaviour when participating in training or coordination in the FL algorithm
- the worker-end customization is done by inhereting the ```WorkerManager``` class and utilizing the 
predefined API's ```register_messafe_receive_handler``` and ```send_message``` to define the receiving 
sending messages without considering the underlying communication mechanism (something the authors
note is not offered by other platforms)

##### Message Definition
- supports message exchange carrying auxiliary information beyond the gradient or model
- ```WorkerManager``` handles messages defined by other trainers and also sends messages defined
by itself
- sending message executed after handling of received message
- can send any message type and related message parameters in the ```train()``` function

##### Topology Management
- ```TopologyManager``` provides a means to manage federated learning system topology and to send
messages to arbitrary neighbours during training
- after ```TopologyManager``` is set, the worker ID of each trainer in the network can be queried,
for example, in addition to other information
- code and more [here](https://github.com/FedML-AI/FedML/tree/master/fedml_core/distributed/topology)
##### Trainer and Coordinator
- The coordinator acts typically as the server and aggregates model updates, more concretely it
coordinates the trainers
- The trainers are typically completing local model training but are simply coordinated by the coordinator
- The developer is free to implement the trainer and coordinator in whichever manner they require
- Different trainer/coordinators have been implemented in the source code already

##### Privacy and Security
- low-level APIs that implement common cryptographic primitives
  - secret sharing
  - key agreement
  - digital signature
  - public key infrastructure
  - Lagrange Coded Computing (LCC; planned)
  - sample implementation of secure aggregation algorithm (planned)
- includes robust aggregation methods
  - KRUM
  - norm difference clipping
  - weak differential private
  - RFA (geometric median)
  - MultiKRUM
### Federated Optimizer
- support a wide variety of federated learning algorithms:
  - FedAvg
  - DecentralizedFL
  - Vertical Federated Learning
  - Split Learning
  - Federated Neural Architecture Search (FedNAS)
  - Turbo-Aggregate
- constantly adding new FL algorithms such as Adaptive Federated Optimizer, FedNova, FedProx, FedMA and more

### Implementation
Complete documentation for the FedML-API and FedML-core APIs can be found at the links below:
- FedML-API: [https://doc.fedml.ai/#/api-fedml-api](https://doc.fedml.ai/#/api-fedml-api)
- FedML-core: [https://doc.fedml.ai/#/api-core](https://doc.fedml.ai/#/api-core)
- FedML-Mobile: [https://doc.fedml.ai/#/api-fedml-mobile](https://doc.fedml.ai/#/api-fedml-mobile)

#### Interpreting Documentation
FedML is not as friendly to new users as other platforms may be as the documentation is actively being
developed. Ample examples are provided in the repository, however. It is hoped that this document can
serve as a introductory guide to fill in the gaps between FedML documentation and source code at 
the time of writing.

#### Standalone FedAVG Example
This sample experiment is provided by FedML's authors in the FedML repository and has been adapted
here for demonstration purposes. Carefully review the provided test.py file and FedML experiment
files in addition to this guide. Some preparation and auxiliary tasks are not discussed here (i.e.
logging, and PyTorch backend preparation).

I recommend referencing either the provided ```test.py``` file or the original ```main_fedavg.py``` file from
the FedML repository for the remainder of this section. ```main_fedavg.py``` can be found [here](https://github.com/FedML-AI/FedML/blob/master/fedml_experiments/standalone/fedavg/main_fedavg.py)

Note: In the following code samples, comments signed with _SB_ are added by the author of this document. 
##### Loading the dataset
A method is defined to load in data from some dataset, typically via a ```load_partition_data_XXXXX()``` 
method found in ```data_loader.py``` of the particular dataset attempted to be loaded from 
_fedml/api/data_preprocessing/XXXXX_. Here's a look at ```load_partition_data_federated_emnist()```
from FedML-API which is called in ```test.py```:
```python
def load_partition_data_federated_emnist(dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE):

    # client ids
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())

    # local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, client_idx)
        local_data_num = len(train_data_local) + len(test_data_local)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
        # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
        #     client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # global dataset
    train_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(train_data_local_dict.values()))
                ),
                batch_size=batch_size, shuffle=True)
    train_data_num = len(train_data_global.dataset)
    
    test_data_global = data.DataLoader(
                data.ConcatDataset(
                    list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
                ),
                batch_size=batch_size, shuffle=True)
    test_data_num = len(test_data_global.dataset)
    
    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique([train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)]))
        logging.info("class_num = %d" % class_num)

    return DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
```

Your implementation may vary greatly, but the
method should return a dataset with this structure:
```python
dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
"""          
           train_data_num   the size of the global training set
            test_data_num   the size of the global test set
        train_data_global   global training dataset (loaded by PyTorch's DataLoader in FedML datasets)
         test_data_global   global test dataset (loaded by PyTorch's DataLoader in FedML datasets)
train_data_local_num_dict   dictionary keys are Client IDs and values are the size of their local dataset
    train_data_local_dict   dictionary keys are Client IDs and values are local training sets
     test_data_local_dict   dictionary keys are Client IDs anf values are local test sets
                class_num   number of output classes
"""
```
Additional reading:
- PyTorch Datasets & DataLoaders: [https://pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- Writing Custom Datasets, DataLoaders and Transforms (PyTorch): [https://pytorch.org/tutorials/beginner/data_loading_tutorial.html](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

Below, is the code in the ```test.py``` which loads in the dataset to be used in standalone simulation.
As mentioned, ```load_data()``` returns the (very large) dataset array which will be used in further
simulation.
```python
# Load data from Federated EMNIST dataset -SB
def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    # For simplicity and easier tracing, I've removed unnecessary options from the original code -SB
    if dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num

    # For simplicity and easier tracing, I've removed unnecessary options from the original code -SB
    else:
        print("[120] Error")
        exit(-1)

    # client_num_in_total != 1 in the test.sh example, so centralized = False -SB
    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset
```
```python
# usage
dataset = load_data(args, args.dataset)
```
##### Model Creation
After loading in the dataset, a model is created using ```create_model(rgs, model_name, output_dim)```
to return either a custom model or one already included in FedML-API. A simplified view of the method
is provided below:
```python
def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    
    if model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)

    return model
```
The method instantiates a class based on ```torch.nn.Module``` with 62 output for Federated Extended
MNIST and returns a ```torch.nn.Module```.

Further reading: 
- ```CNN_Dropout``` class definition [here](https://github.com/FedML-AI/FedML/blob/2ee0517a7fa9ec7d6a5521fbae3e17011683eecd/fedml_api/model/cv/cnn.py#L74)
- ```torch.nn.Module``` class documentation [here](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

##### Model Training Setup
The model can now be trained with one convenient line that calls ```custom_model_trainer()```.
This method is defined ```test.py``` and initializes a ```MyModelTrainerTAG``` object. The 
```custom_model_trainer()``` definition is shown below.

```python
def custom_model_trainer(args, model):
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)
```
The ```MyModelTrainer``` is imported as the ```MyModelTrainerTAG```, and is defined in ```fedml_api/standalone/fedavg/my_model_trainer_tag_prediction.py```
which is an abstraction of the the ```ModelTrainer``` class in ```fedml_core/trainer/model_trainer.py``` 
and facilitates the training of a PyTorch model. For reference, the ```MyModelTrainer``` 
class definition is shown below.

```python
class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.BCELoss(reduction='sum').to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        criterion = nn.BCELoss(reduction='sum').to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                predicted = (pred > .5).int()
                correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                true_positive = ((target * predicted) > .1).int().sum(axis=-1)
                precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                recall = true_positive / (target.sum(axis=-1) + 1e-13)
                metrics['test_precision'] += precision.sum().item()
                metrics['test_recall'] += recall.sum().item()
                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
```

```python
# usage
model_trainer = custome_model_trainer(args, model)
```

##### FedAVG Training
Once the model_trainer has been created from the model, it is passed with the dataset and other 
parameters to initialize a new ```FedAvgAPI``` object which is then trained with two lines:

```python
fedAvgAPI = FedAvgAPI(dataset, device, args, model_trainer)
fedavgAPI.train()
```
It is important to note that device is a PyTorch device defined as: 
```python
device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
```
The ```FedAvgAPI``` class is defined in ```fedml_api/standalone/fedavg/fedavg_api.py``` and handles
client setup, training, sampling, validation, aggregation. After being initialized, the ```FedAvgAPI```
method, ```train()```, is called to begin the federated learning simulation.

For reference, the ```train()``` definition is included below:
```python
 def train(self):
    w_global = self.model_trainer.get_model_params()
    for round_idx in range(self.args.comm_round):

        logging.info("################Communication round : {}".format(round_idx))

        w_locals = []

        """
        for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
        Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
        """
        client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
        logging.info("client_indexes = " + str(client_indexes))

        for idx, client in enumerate(self.client_list):
            # update dataset
            client_idx = client_indexes[idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # train on new dataset
            w = client.train(copy.deepcopy(w_global))
            # self.logger.info("local weights = " + str(w))
            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

        # update global weights
        w_global = self._aggregate(w_locals)
        self.model_trainer.set_model_params(w_global)

        # test results
        # at last round
        if round_idx == self.args.comm_round - 1:
            self._local_test_on_all_clients(round_idx)
        # per {frequency_of_the_test} round
        elif round_idx % self.args.frequency_of_the_test == 0:
            if self.args.dataset.startswith("stackoverflow"):
                self._local_test_on_validation_set(round_idx)
            else:
                self._local_test_on_all_clients(round_idx)
```

#### Testing
To run the test experiment from this folder, you will need to have FedML installed correctly as per
the instructions included above. Following this, you can simply run the follow shell script:
```shell
sh test.sh 0 10 10 20 femnist ./../../../data/FederatedEMNIST/datasets cnn hetero 1000 1 0.03 sgd 0
```

([test.sh source](https://github.com/FedML-AI/FedML/blob/master/fedml_experiments/standalone/fedavg/run_fedavg_standalone_pytorch.sh))

Note: You may need to change the location of ```./../../../data/FederatedEMNIST/datasets``` to point
as required by your system. Furthermore, this provided example will not allow you to change the dataset
used in the experiment and may limit not work if parameters are changed. Please see the test.sh 
source for a more complete example.
## The Bottom Line
Even though FedML seems to focus heavily on extensibility and making it easy to deploy any federated
learning topology to near any environment in a manner that is easy and scalable, I personally have not
found it as easy to use as other platforms. Primarily, this is due to a steep learning curve and, in
my opinion, not very complete or detailed documentation. There does not appear to be any clean and
quick tutorials that don't just leave how the code functions to the interpretation of the data scientist. 
This adds a large initial time investment to using the platform, especially to those less experienced.
Furthermore, FedML does not seem to natively support simulation using virtual machines (or networks 
or PCs for that matter).

