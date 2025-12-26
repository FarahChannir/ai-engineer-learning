import numpy as np
import matplotlib.pyplot as plt

# House sizes in square meters (40 samples)
x = np.array([
     55,  60,  65,  70,  75,  80,  85,  90,  95, 100,
    105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
    155, 160, 165, 170, 175, 180, 185, 190, 195, 200,
    210, 220, 230, 240, 250, 260, 270, 280, 290, 300
])

# Corresponding house prices (in USD or TRY for example)
y = np.array([
     75000,  80000,  85000,  90000,  95000, 100000, 107000, 113000, 118000, 124000,
    128000, 134000, 138000, 142000, 148000, 153000, 157000, 163000, 168000, 173000,
    177000, 183000, 189000, 193000, 198000, 204000, 209000, 214000, 219000, 225000,
    235000, 245000, 255000, 265000, 275000, 285000, 295000, 305000, 315000, 325000
])

# Initialize parameters
w = 2000 # weight (slope)
b = 100000  # bias (intercept)

# Hyperparameters
alpha =0.0000001 # learning rate
iterations = 4000
m = len(x)

# Gradient Descent
for i in range(iterations):
    y_pred = w * x + b

    dw = (1/m) * np.sum((y_pred - y) * x)
    db = (1/m) * np.sum(y_pred - y)

    w -= alpha * dw
    b -= alpha * db

    if i % 200 == 0:
        cost = (1/(2*m)) * np.sum((y_pred - y)**2)
        print(f"Iteration {i}: Cost={cost:.2f}, w={w:.2f}, b={b:.2f}")

print("\nFinal model:")
print(f"Price = {w:.2f} * Size + {b:.2f}")

# Plot the data and fitted line
plt.scatter(x, y, color='blue', label='Data (House Prices)')
plt.plot(x, w*x + b, color='red', label='Fitted Line')
plt.xlabel("House Size (m²)")
plt.ylabel("House Price")
plt.title("House Price Prediction using Gradient Descent")
plt.legend()
plt.show()
