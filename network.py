import numpy as np

# Networks starts off by doing a forward pass of the first input to get the initial activations
# and results. When it starts, our activations and raw sums are null. They're then given values after the forward run
# which is updated after each pass, together with the weights. That's how the training/learning occurs

class Network(object):
    def __init__(self):
        self.z1 = None
        self.z2 = None
        self.z3 = None

        self.a1 = None
        self.a2 = None
        self.a3 = None

        self.one_hot = None

        self.w1 = np.random.uniform(-1, 1, (748, 16))
        self.w2 = np.random.uniform(-1, 1, (16, 16))
        self.w3 = np.random.uniform(-1, 1, (16, 10))

        self.b1 = np.random.uniform(-1, 1, 16)
        self.b2 = np.random.uniform(-1, 1, 16)
        self.b3 = np.random.uniform(-1, 1, 10)

        self.lr = 0.01

    def forward(self, neurons):
        # multiply the weights by the inputs, plus the bias, then sum them all up
        self.z1 = np.dot(neurons, self.w1) + self.b1
        self.a1 = np.maximum(0, self.z1)

        # move on to the next weights and activations (second layer)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = np.maximum(0, self.z2)

        # final results use softmax. Turns all 10 results into probabilities that sum
        # up to 1
        self.z3 = np.dot(self.a2, self.w3) + self.b3

        def softmax(x):
            # Subtracting the max (numerical stability)
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        self.a3 = softmax(self.z3)

        return self.a3



    # Takes in the prediction from the forward pass, and outputs a single number
    # using the loss function (Mean Squared Error) to find the average squared
    # difference between the predicted values and the actual values
    # E.g.

    def get_loss(self, label):
        self.one_hot = np.zeros(10)
        self.one_hot[label] = 1

        # Now find MSE
        mse = np.mean(np.square(self.a3 - self.one_hot))

        return mse


    def backward_prop(self, x):

        def calc_weight(dl_da, activation):
            return np.dot(activation.T, dl_da)

        def calc_activation(dl_da, weights):
            return np.dot(dl_da, weights.T)

        dl_da3 = self.a3 - self.one_hot
        dl_dw3 = calc_weight(dl_da3, self.a2)

        dl_da2 = calc_activation(dl_da3, self.w3)
        dl_dw2 = calc_weight(dl_da2, self.a1)

        dl_da1= calc_activation(dl_da2, self.w2)

        dl_dw1 = calc_weight(dl_da1, x)




