import numpy as np
import matplotlib.pyplot as plt

# Example data: house size (m²) and price ($1000s)
x = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
y = np.array([150, 180, 200, 220, 240, 260, 275, 290, 310, 330])

# Initialize parameters
w = 2.0  # slope
b = 50.0 # intercept
alpha = 0.0001  # learning rate

# Perform a few gradient descent steps manually
for i in range(1000):
    y_pred = w * x + b
    dw = (-2 / len(x)) * np.sum(x * (y - y_pred))
    db = (-2 / len(x)) * np.sum(y - y_pred)
    w = w - alpha * dw
    b = b - alpha * db

# Plot the data points
plt.scatter(x, y, color='blue', label='Training data')

# Plot the regression line
y_pred = w * x + b
plt.plot(x, y_pred, color='red', label=f'Regression line (y = {w:.2f}x + {b:.2f})')

# Add labels and legend
plt.xlabel('House Size (m²)')
plt.ylabel('House Price ($1000s)')
plt.title('Linear Regression: House Size vs. Price')
plt.legend()
plt.grid(True)
plt.show()
