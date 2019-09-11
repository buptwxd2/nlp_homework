import numpy as np

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


# define Input class
class Input(Node):
    # there is no inputs for Input Nodes
    def __init__(self, name, ini_value):
        Node.__init__(self, name, inputs=[])  # no inputs for Input Nodes
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

        # TODO: here i want to check how many outupt nodes
        # need to think twice    ??
        # for matrix: D = W * X => dW = dD * X.T, dX = W.T * dD
        # for matrix: Y = W * X => dW = dY * X.T, dX = W.T * dY
        # ToDO: check the gradient formula, first need to check their dimension or shape
        self.gradients[self.x_node] = np.dot(grad_cost, self.w_node.value.T)
        self.gradients[self.w_node] = np.dot(grad_cost, self.w_node.value.T)


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
        #TODO Here i want to check the shape of y_true and y_hat
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

