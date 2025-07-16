import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")
X = data['Size'].values
y = data['Price'].values

plt.scatter(X, y, color='r', marker='x')
plt.xlabel("House size (m2)")
plt.ylabel("House price ($)")
plt.title("House Size vs. Price")
plt.grid(True)
plt.show()

def compute_cost(X, y, w, b):
    m=len(X)
    cost=0.0
    for i in range(m):
        f_wb=w*X[i]+b
        cost += (f_wb-y[i])**2
    total_cost = cost/(2*m)
    return total_cost

def compute_gradient(X, y, w, b):
    m=len(X)
    dj_dw=0.0
    dj_db=0.0
    for i in range(m):
        f_wb=w*X[i]+b
        dj_dw += (f_wb-y[i])*X[i]
        dj_db += (f_wb-y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w=w_in
    b=b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w=w-alpha*dj_dw
        b=b-alpha*dj_db

        if i%100==0 or i==num_iters-1:
            cost = compute_cost(X, y, w, b)
            print(f"Iteration {i:4}: w = {w:.2f}, b = {b:.2f}, cost = {cost:.2f}")
    return w, b
# treniranje modela

initial_w=0
initial_b=0
alpha=0.0001
iterations=1000

w_final, b_final = gradient_descent(X,y,initial_w,initial_b,alpha,iterations)
print(f"Trained parameters: w = {w_final:.2f}, b = {b_final:.2f}")

plt.scatter(X, y, color='r', marker='x', label="Training data")
plt.xlabel("House size (m2)")
plt.ylabel("House price ($)")
plt.title("House Size vs. Price")
x_vals = np.array([min(X), max(X)])
y_vals=w_final*x_vals+b_final
plt.plot(x_vals,y_vals,color='b', label="Regression line")
plt.grid(True)
plt.legend()
plt.savefig("regression_plot.png")
plt.show()

velicina_test = 110
predikcija=w_final*velicina_test + b_final
print(f"Prediction of a house price, size: {velicina_test} m2 is {predikcija:.2f} $.")
