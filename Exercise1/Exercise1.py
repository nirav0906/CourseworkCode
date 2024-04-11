import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# Load data from Excel file
df = pd.read_excel('Exercise1Data.xlsx')

# Define the dependent variable and the independent variables

Y = df['Demand']

X1 = df[['Temp']]
X1 = sm.add_constant(X1)

X2 = df[['Temp', 'Hour','Hour2','Hour3']]
X2 = sm.add_constant(X2)  # Add a constant term to the independent variables


# Model 1
model1 = sm.OLS(Y, X1)
results1 = model1.fit()

# Model 2
model2 = sm.OLS(Y, X2)
results2 = model2.fit()

print(results1.summary())
print(results2.summary())


print('Model 1:')
print('Beta0 (intercept):', results1.params.iloc[0])
print('Beta1 (slope):', results1.params.iloc[1])

print('t-values:', results1.tvalues.values)
print('p-values:', results1.pvalues.values)
print('R-squared:', results1.rsquared)

print('Model 2:')
print('Beta0 (intercept):', results2.params.iloc[0])
print('Beta1 (slope):', results2.params.iloc[1])

print('t-values:', results2.tvalues.values)
print('p-values:', results2.pvalues.values)
print('R-squared:', results2.rsquared)



# Plot for Model 1
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['Temp'], Y, label='Data')
plt.plot(df['Temp'], results1.fittedvalues, color='red', label='OLS')
plt.xlabel('Temp')
plt.ylabel('Demand')
plt.title('Model 1: Demand vs Temp')
plt.legend()

# Plot for Model 2
# Since Model 2 is a multiple regression model, we can't plot it on a 2D plot.
# We can plot the actual vs predicted values instead.
plt.subplot(1, 2, 2)
plt.scatter(Y, results2.fittedvalues)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Model 2: Actual vs Predicted Demand')

plt.tight_layout()
plt.show()