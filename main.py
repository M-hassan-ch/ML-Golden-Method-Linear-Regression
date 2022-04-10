import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def find_coefficients(x, y):
    # finding mean
    x_mean = (x.sum() / x.size)
    y_mean = (y.sum() / y.size)

    num = ((x - x_mean) * (y - y_mean))  # nominator
    den = ((x - x_mean) ** 2)  # denominator

    slope = (num.sum() / den.sum())
    intercept = y_mean - (slope * x_mean)  # this is derived by re-arranging y=mx+c -> c=y-mx

    return slope, intercept


def cal_error(y, y_pred):
    y_mean = (y.sum() / y.size)

    num = (y_pred - y_mean) ** 2
    den = (y - y_mean) ** 2

    return num.sum() / den.sum()


########################################################

if __name__ == '__main__':

    df = pd.DataFrame(pd.read_csv("Salary_Data.csv"))

    x = np.array(df.iloc[:, 0])  # Experience of employee
    y = np.array(df.iloc[:, 1])  # Salary of employee

    slope, intercept = find_coefficients(x, y)
    y_pred = ((slope * x) + intercept)    # Predicting values using farmula

    # visualizing results      ( plotting results derived from least square method )

    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.title("Using least square method")
    plt.xlabel("No. of years")
    plt.ylabel("Salary")
    plt.plot(x, y_pred, marker='o', color='blue')

    print("\nUsing least square method\n")
    print("Calculated slope: - ", slope)
    print("Calculated intercept: - ", intercept)
    print("Error through point-line: - ", cal_error(y, y_pred))

    # using sklearn

    reg = LinearRegression()

    reg = reg.fit(x.reshape(-1, 1), y)
    print("\nUsing sklearn")
    print("\nCalculated slope: - ", reg.coef_)
    print("Calculated intercept: - ", reg.intercept_)
    print("Error through sklearn: - ", reg.score(x.reshape(-1, 1), y))
    print("\nError difference: - ", reg.score(x.reshape(-1, 1), y)-cal_error(y, y_pred))

    # visualizing results

    plt.subplot(1, 2, 2)                # ( plotting results derived from sklearn )
    plt.scatter(x, y)
    plt.title("Using sklearn library")
    plt.xlabel("No. of years")
    plt.ylabel("Salary")
    plt.plot(x, reg.predict(x.reshape(-1,1)), marker = 'o', color = 'r')
    plt.show()

