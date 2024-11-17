import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
X = np.array([[1, 2], [0, 1], [-1, 2], [0, -1], [-1, -2], [-2, 0]])  # Feature vectors (n_samples x n_features)
y = np.array([1, 1, 1, -1, -1, -1])  # Labels (+1 or -1)

# Compute A matrix
n_samples = len(y)
A = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        A[i, j] = y[i] * y[j] * np.dot(X[i], X[j])

# Objective function
def objective(alpha):
    alpha = np.array(alpha)
    term1 = -0.5 * np.dot(alpha.T, np.dot(A, alpha))
    term2 = np.sum(np.abs(alpha))  # L1 norm
    return -(term1 + term2)


bounds = Bounds(0, np.inf)

# Initial guess for alpha
alpha0 = np.zeros(n_samples)

# Solve the optimization problem
result = minimize(objective, alpha0, method="SLSQP", bounds=bounds)  # SLSQP handles constraints well
alpha_optimal = result.x

w = np.sum(alpha_optimal[:, None] * y[:, None] * X, axis=0)
support_vector_idx = np.where(alpha_optimal > 1e-5)[0][0]  # Find the first support vector
b = y[support_vector_idx] - np.dot(w, X[support_vector_idx])
# Output psi^0
print("w:", w)
print("b:", b)

# Plot the data points
plt.figure(figsize=(8, 6))
for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='Class +1' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Class -1' if i == 0 else "")

# Create a mesh grid for the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = w[0] * xx + w[1] * yy + b

# Plot the decision boundary
plt.contour(xx, yy, Z, levels=[0], colors='green', linestyles='--')

# Labels and legend
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.legend()
plt.grid()
plt.savefig("svm_ls.png")