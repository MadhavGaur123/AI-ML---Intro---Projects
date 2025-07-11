import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
print("Opening CSV and reading CSV file")
data = pd.read_csv(r"C:\Users\gaurm\Downloads\house-prices.csv", encoding='latin-1')
X = data[["SqFt" , "Bedrooms"]]
Y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42) #split the data into training set and testing set test_size = 0.2 split the data into 80/20 ratio and random_state = 42 tells parameter to ensure reproducibility
print(np.size(X))
print(np.size(X_train))
print(np.size(X_test))
machine = LinearRegression()
machine.fit(X_train,y_train)
slope = machine.coef_[0]
intercept = machine.intercept_
print("Slope: ", slope)
print("intercept: ", intercept)
y_pred = machine.predict(X_test)
MSE = mean_squared_error(y_test,y_pred)
r = r2_score(y_test,y_pred)
print(MSE)
threshold_r_value = 0.5
if(r<=threshold_r_value):
    print("The machine is not predicting well enough. Please increase the data size")
    print(r)
else:
    print("The machine is giving correct prediction with ample scope of error")
    print(r)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the actual data
ax.scatter(X["SqFt"], X["Bedrooms"], Y, color='blue', alpha=0.5, label='Actual Data')

# Create a meshgrid for SqFt and Bedrooms to plot the regression plane
x_surf, y_surf = np.meshgrid(np.linspace(X["SqFt"].min(), X["SqFt"].max(), 100),
                             np.linspace(X["Bedrooms"].min(), X["Bedrooms"].max(), 100))

# Predict the corresponding prices using the linear regression model
z_surf = machine.predict(np.c_[x_surf.ravel(), y_surf.ravel()])
z_surf = z_surf.reshape(x_surf.shape)

# Plot the regression plane
ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.3, rstride=100, cstride=100)

# Labels and title
ax.set_title("Linear Regression: House Prices vs. SqFt and Bedrooms")
ax.set_xlabel("Square Feet (SqFt)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price ($)")
ax.legend()

plt.show()
