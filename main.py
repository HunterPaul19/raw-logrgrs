from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleLogisticRegression:
    def __init__(self, learning_rate = 0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0

    def sigmoid(self, x):
        # Sigmoid Activation Function
        return 1 / (1 + np.exp(-x))

    def cost (self, h, y):
        ## Cross-entropy loss
        m = len(y)
        return (-1/m) * np.sum(y * np.log(h) + (1-y) * np.log(1 - h))

    def fit(self, X, y):
        # Train model with gradient descent
        m, n = X.shape
        self.weights = np.zeros(n)

        self.cost_history = []

        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)

            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.cost_history.append(self.cost(h, y))

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights) + self.bias) >= 0.5).astype(int)

np.random.seed(0)
X = np.random.rand(200, 2) * 10
y = (X[:, 0] + X[:, 1] > 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SimpleLogisticRegression(learning_rate=0.1, n_iters=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Model Accuracy: {accuracy:.2f}")

print(model.weights)
print(model.bias)

plt.plot(model.cost_history)
plt.title("Cost Function Convergance")
plt.xlabel("Iterations")
plt.grid(True)
plt.show()
    

    