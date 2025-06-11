import numpy as np

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# network dimensions
input_size = 2
hidden_size = 4
output_size = 1

# initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
B1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
B2 = np.zeros((1, output_size))

# activation functions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

lr = 0.1
epochs = 10000

for epoch in range(epochs):
    # forward
    Z1 = X.dot(W1) + B1
    A1 = np.tanh(Z1)
    Z2 = A1.dot(W2) + B2
    A2 = sigmoid(Z2)
    
    # loss (binary cross entropy)
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))
    
    # backward
    dZ2 = A2 - y
    dW2 = A1.T.dot(dZ2) / len(X)
    dB2 = np.mean(dZ2, axis=0, keepdims=True)
    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * tanh_deriv(Z1)
    dW1 = X.T.dot(dZ1) / len(X)
    dB1 = np.mean(dZ1, axis=0, keepdims=True)
    
    # update
    W2 -= lr * dW2
    B2 -= lr * dB2
    W1 -= lr * dW1
    B1 -= lr * dB1
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# final predictions
preds = (A2 > 0.5).astype(int)
print("Predictions after training:")
for x, pred in zip(X, preds):
    print(f"Input: {x}, Predicted: {pred[0]}")
