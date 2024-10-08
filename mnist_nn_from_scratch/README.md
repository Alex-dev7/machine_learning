### Neural Network for MNIST Classification
This project is a simple implementation of a 2-layer neural network for classifying handwritten digits from the MNIST dataset.

##### Project Structure
The main script is nn_for_MNIST.py, which contains the implementation of the neural network.

##### How it Works
The neural network is implemented in Python using NumPy for numerical computations. The network has two layers: a hidden layer and an output layer. The hidden layer uses the ReLU activation function, and the output layer uses the softmax function for multi-class classification.

The network is trained using gradient descent. The cost function is the cross-entropy loss, which is suitable for classification problems.

