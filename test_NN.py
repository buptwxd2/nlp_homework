import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt

class Node:
    """
    Methods:
        forward
        backward
    Attributes:
        gradient
        inputs
        outputs
    """
    def __init__(self, name='', inputs=[]):
        self.name = name
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        # Build the connection between nodes
        # we feed the input nodes for one node and append output nodes in its successors
        for node in self.inputs:
            node.outputs.append(self)  # build the connection relationship

    def forward(self):
        """Forward propagation

        compute the output value based on input nodes and store the value
        into *self.value*
        """
        raise NotImplemented

    def backward(self):
        """ Back propagation

        compute the gradient of each input node and store the value
        into "self.gredients"
        """
        raise NotImplemented

    def __repr__(self):
        return "Node: {}".format(self.name)


# define Input class
class Input(Node):
    # there is no inputs for Input Nodes
    def __init__(self, name):
        Node.__init__(self, name, inputs=[])  # no inputs for Input Nodes

    def get_init_value(self, ini_value):
        self.ini_value = ini_value

    def forward(self):
        self.value = self.ini_value

    def backward(self):
        grad_cost = 0
        for output_node in self.outputs:
            grad_cost += output_node.gradients[self]

        self.gradients[self] = grad_cost

    def __repr__(self):
        return "Input node: {}".format(self.name)


# define Linear class
class Linear(Node):
    def __init__(self, name, x, weights, bias):
        Node.__init__(self, name=name, inputs=[x, weights, bias])
        self.x_node = x
        self.w_node = weights
        self.b_node = bias

    def forward(self):
        self.value = np.dot(self.x_node.value, self.w_node.value) + self.b_node.value

    def backward(self):
        grad_cost = 0
        for output_node in self.outputs:
            grad_cost += output_node.gradients[self]

        # for matrix: Y = X * W + b => dW = X.T * dY, dX = dY * W.T, db = np.sum(dY, axis=0, keepdims=False)
        self.gradients[self.w_node] = np.dot(self.x_node.value.T, grad_cost)
        self.gradients[self.x_node] = np.dot(grad_cost, self.w_node.value.T)
        self.gradients[self.b_node] = np.sum(grad_cost, axis=0, keepdims=False)


# define Sigmoid class
class Sigmoid(Node):
    def __init__(self, name, node):
        Node.__init__(self, name=name, inputs=[node])
        self.x_node = node

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.x_node.value)

    def backward(self):
        grad_cost = 0
        for output_node in self.outputs:
            grad_cost += output_node.gradients[self]

        self.gradients[self.x_node] = grad_cost * (self.value * (1 - self.value))


class MSE(Node):
    def __init__(self, name, y_true, y_hat):
        Node.__init__(self, name=name, inputs=[y_true, y_hat])
        self.y_true = y_true
        self.y_hat = y_hat

    def forward(self):
        y_true_flatten = self.y_true.value.reshape(-1, 1)
        y_hat_flatten = self.y_hat.value.reshape(-1, 1)
        self.y_diff = y_true_flatten - y_hat_flatten

        self.value = np.mean(self.y_diff ** 2)

    def backward(self):
        n = self.y_true.value.shape[0]

        self.gradients[self.y_true] = 2/n * self.y_diff
        self.gradients[self.y_hat] = -2/n * self.y_diff


def train_one_batch(topological_sorted_graph):
    # perform forward along with the list, simulating data flow
    for node in topological_sorted_graph:
        node.forward()

    # then backward to calculate the gradient in a reverse direction
    for node in topological_sorted_graph[::-1]:
        # save the gradients in the Node property: gradients
        node.backward()


# Linear -> Sigmoid -> MSE
def topological_sort(feed_dict):
    # starting with input nodes, since input nodes are in feed_dict
    input_nodes = [node for node in feed_dict.keys()]

    starting_nodes = input_nodes[:]   # or     starting_nodes = list(input_nodes)

    G = {}
    while starting_nodes:
        node = starting_nodes.pop(0)
        if node not in G:
            G[node] = {"in": set(), "out": set()}

        for output_node in node.outputs:
            if output_node not in G:
                G[output_node] = {"in": set(), "out": set()}

            G[node]['out'].add(output_node)
            G[output_node]['in'].add(node)
            starting_nodes.append(output_node)

    # G = {}
    # # traverse all nodes in G
    # from_input_nodes = list(input_nodes)
    # all_nodes = list(input_nodes)
    # while from_input_nodes:
    #     node = from_input_nodes.pop(0)
    #     for output_node in node.outputs:
    #         if output_node not in from_input_nodes:
    #             from_input_nodes.append(output_node)
    #
    #         if output_node not in all_nodes:
    #             all_nodes.append(output_node)
    # # Filling G
    # for any_node in all_nodes:
    #     G[any_node] = {"in": any_node.inputs,
    #                    "out": any_node.outputs}

    L = []
    # starting with input nodes, S records nodes whose inputs are None
    S = set(input_nodes)
    while S:
        node = S.pop()

        # Special handling input Nodes, need to feed the input value to input Nodes
        # if isinstance(node, Input):
        #     node.value = feed_dict[node]

        L.append(node)
        for output_node in node.outputs:
            G[output_node]["in"].remove(node)
            # not sure if the below line is necessary
            G[node]["out"].remove(output_node)

            if len(G[output_node]["in"]) == 0:
                S.add(output_node)

    # sorted topological node list
    return L


def sgd_update(trainable_nodes, learning_rate=1e-2):
    for node in trainable_nodes:
        # Only input node has node.gradients per itself
        node.value += -1 * learning_rate * node.gradients[node]


"""
Node connection in this graph =>(Sort topology) Get sorted topological graph
=> Run train_one_batch to forward and backward, getting the gradients
=> Run sgd_update to update the variable node values
=> Repeat train_one_batch and sgd_update until the loss is small enough => break
"""

# (Input) -> Linear -> Sigmoid -> Linear -> MSE

# 1. Build the Nodes in this graph
X, y = Input(name='X'), Input(name='y')   # tensorflow -> placeholder
W1, b1 = Input(name='W1'), Input(name='b1')
W2, b2 = Input(name='W2'), Input(name='b2')

# 2. Build Node connections in this graph
linear_output = Linear("linear_node_1", X, W1, b1)
sigmoid_output = Sigmoid("sigmoid_node", linear_output)
y_hat = Linear("linear_node_2", sigmoid_output, W2, b2)
loss = MSE("mse_node", y, y_hat)

# 3. Feed the input with values and sort topological
#  3.1 load X,y from sklearn
from sklearn.datasets import load_boston
data = load_boston()
X_ = data['data']
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)
n_features = X_.shape[1]
y_ = data['target']

n_hidden = 10  # number of neurons in the first layer

W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

print("X_shape: {}".format(X_.shape))
print("y_shape: {}".format(y_.shape))

#  feed into X,Y and W,b
feed_dict ={
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}
graph = topological_sort(feed_dict)
W1.get_init_value(W1_)
b1.get_init_value(b1_)
W2.get_init_value(W2_)
b2.get_init_value(b2_)
print(graph)


# 3. Begin to forward and get the final loss; Then backward to update the parameters
losses = []
epochs = 5000
learning_rate = 1e-3

batch_size = 64
steps_per_epoch = X_.shape[0] // batch_size

for epoch in range(epochs):
    epoch_loss = 0

    for batch in range(steps_per_epoch):
        X_batach, y_batch = resample(X_, y_, n_samples=batch_size)
        X.get_init_value(X_batach)
        y.get_init_value(y_batch)

        # Run forward => get the loss; Then Run backward => get the gradients
        train_one_batch(graph)
        # update the parameters per gradients
        sgd_update(trainable_nodes=[W1, b1, W2, b2], learning_rate=learning_rate)

        # Record current loss
        epoch_loss += loss.value

    losses.append(epoch_loss/steps_per_epoch)

    if epoch % 100 == 0:
        print("Epoch: {}, loss={:.3f}".format(epoch, epoch_loss/steps_per_epoch))

plt.plot(losses)
plt.show()

print("Optimal parameters:")
print(W1.value)
print(b1.value)
print(W2.value)
print(b2.value)


