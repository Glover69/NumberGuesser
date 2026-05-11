# NumberGuesser

This project demonstrates how neural networks work at their core, built from scratch using only NumPy, with no machine learning libraries.

Using the MNIST dataset of handwritten digits and their labels, the network learns to correctly identify digits 0–9 from a 28×28 pixel image.

---

## How does it work?

A neural network is made up of neurons organised into layers, with weights connecting every neuron in one layer to every neuron in the next.

### Setup

- **Weights** are initialised with random values in the range (-1, 1). There are 3 weight matrices connecting our 4 layers.
- **Biases** are also initialised randomly in the same range. They act as a threshold, controlling how easily a neuron activates independently of its inputs.
- **The input layer** is formed from the image itself each image is a 28×28 grayscale image, giving us 784 neurons. Each neuron's activation value represents how bright or dark that pixel is, normalised to a range of 0–1.
- **Two hidden layers** of 128 neurons each sit between the input and output.
- **The output layer** has 10 neurons one for each digit (0–9). The neuron with the highest activation is the network's guess.

---

## Training process

Training is a repeated loop over every image in the dataset. For each image:

1. **Forward pass** — the image is passed through the network layer by layer to produce a prediction
2. **Loss calculation** — the prediction is compared to the correct answer to get a single number representing how wrong the network was
3. **Backpropagation** — working backwards through the network, gradients are computed for every weight and bias, telling us how much each one contributed to the loss
4. **Weight update** — every weight and bias is nudged in the direction that reduces the loss

One full pass through the entire dataset is called an **epoch**. The network is trained over multiple epochs, with the data shuffled each time.

---

## Methods

### `forward(image)`
Takes in a flattened 784-value image and passes it through the network layer by layer.

Each layer computes:
```
z = np.dot(input, weights) + bias
a = activation_function(z)
```

What we're doing here is multiplying each neuron by its weight and adding the bias. We get `z` and pass it through **ReLU** as the activation function, returning the value if positive, otherwise 0. The output layer uses **Softmax**, which converts the 10 raw values into probabilities that sum to 1.

Returns a vector of 10 probabilities, one per digit.

---

### `get_loss(label)`
The loss of the prediction (how wrong it is) is obtained by comparing the network's prediction to the correct label using **Mean Squared Error (MSE)**:

```
loss = mean((prediction - expected)²)
```

The expected output is a one-hot vector, e.g. for the digit 3: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

This tells us how far off we are from the correct answer, represented as a single number.

---

### `backward_prop(image)`
The core of training. Works backwards through the network using the **chain rule** to compute how much each weight and bias contributed to the loss.

Starting from the output:
```
dL/da3 = a3 - one_hot
```

`dL/da3` represents the small change in loss with respect to the change in activation. We need to find this, and the equivalent for the weights, for every layer, that's `dL/da` and `dL/dw` for each layer.

Working backwards:
- **Weight gradient** = `activation_in.T · dL/da`  (used to update the weight)
- **Activation gradient** = `dL/da · weights.T`  (passed back to the previous layer)

After computing all gradients, weights and biases are updated:
```
weight = weight - learning_rate × gradient
```

---

## Results

We started with basic logic, one training session, and no bias updates, scoring between **75–77%** accuracy.

Adding bias updates didn't make much difference on its own, nudging us to **~78%**.

Increasing the hidden layer neurons from **16 to 128** made the biggest difference, jumping accuracy to **86.67%**.

Introducing **5 epochs** pushed it further to **88.66%**, which felt like the ceiling for this architecture.

| Configuration | Accuracy |
|---|---|
| 16 hidden neurons, no bias updates | ~77% |
| 16 hidden neurons, with bias updates | ~78% |
| 128 hidden neurons, with bias updates | ~87% |
| 128 hidden neurons, 5 epochs | ~88.66% |

---

## What's next

All further experiments will be on separate branches, with the core implementation staying in `main`.

- [ ] Test with custom handwritten digits
- [ ] Experiment with deeper architectures
- [ ] Implement proper weight initialisation (Xavier/He)
- [ ] Rebuild in PyTorch