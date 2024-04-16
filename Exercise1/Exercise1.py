import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})


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

# print(results1.summary())
# print(results2.summary())


# print('Model 1:')
# print('Beta0 (intercept):', results1.params.iloc[0])
# print('Beta1 (slope):', results1.params.iloc[1])

# print('t-values:', results1.tvalues.values)
# print('p-values:', results1.pvalues.values)
# print('R-squared:', results1.rsquared)

# print('Model 2:')
# print('Beta0 (intercept):', results2.params.iloc[0])
# print('Beta1 (slope):', results2.params.iloc[1])

# print('t-values:', results2.tvalues.values)
# print('p-values:', results2.pvalues.values)
# print('R-squared:', results2.rsquared)



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

# plt.tight_layout()
# plt.show()

WinterWeek = df.iloc[2209:2376] # 1 week of data in winter from 1 Jan to 7 Jan
SummerWeek = df.iloc[7321:7488] # 1 week of data in summer from 1 Aug to 7 Aug

# Predictions for Winter Week
X1_WinterWeek = WinterWeek[['Temp']]
X1_WinterWeek = sm.add_constant(X1_WinterWeek)
Y_WinterWeek = WinterWeek['Demand']
predictions1_WinterWeek = results1.predict(X1_WinterWeek)

# Predictions for Summer Week
X1_SummerWeek = SummerWeek[['Temp']]
X1_SummerWeek = sm.add_constant(X1_SummerWeek)
Y_SummerWeek = SummerWeek['Demand']
predictions1_SummerWeek = results1.predict(X1_SummerWeek)

plt.figure(figsize=(16, 6))
plt.subplot(2, 2, 1)
plt.plot(WinterWeek.index, Y_WinterWeek, label='Actual')
plt.plot(WinterWeek.index, predictions1_WinterWeek, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 1: Winter Week')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(SummerWeek.index, Y_SummerWeek, label='Actual')
plt.plot(SummerWeek.index, predictions1_SummerWeek, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 1: Summer Week')
plt.legend()



# Model 2
# Predictions for Winter Week
X2_WinterWeek = WinterWeek[['Temp','Hour','Hour2','Hour3']]
X2_WinterWeek = sm.add_constant(X2_WinterWeek)
Y_WinterWeek = WinterWeek['Demand']
predictions2_WinterWeek = results2.predict(X2_WinterWeek)

# Predictions for Summer Week
X2_SummerWeek = SummerWeek[['Temp','Hour','Hour2','Hour3']]
X2_SummerWeek = sm.add_constant(X2_SummerWeek)
Y_SummerWeek = SummerWeek['Demand']
predictions2_SummerWeek = results2.predict(X2_SummerWeek)

plt.subplot(2, 2, 3)
plt.plot(WinterWeek.index, Y_WinterWeek, label='Actual')
plt.plot(WinterWeek.index, predictions2_WinterWeek, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 2: Winter Week')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(SummerWeek.index, Y_SummerWeek, label='Actual')
plt.plot(SummerWeek.index, predictions2_SummerWeek, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 2: Summer Week')
plt.legend()

# plt.tight_layout()
# plt.show()




# Temperature vs Demand

plt.figure(figsize=(16, 6))
plt.subplot(2, 2, 1)
plt.plot(WinterWeek.index, WinterWeek['Temp'], label='Temperature')
plt.plot(WinterWeek.index, predictions1_WinterWeek, color='red', label='Demand')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 1: Winter Week')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(SummerWeek.index, SummerWeek['Temp'], label='Temperature')
plt.plot(SummerWeek.index, predictions1_SummerWeek, color='red', label='Demand')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 1: Summer Week')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(WinterWeek.index, WinterWeek['Temp'], label='Temperature')
plt.plot(WinterWeek.index, predictions2_WinterWeek, color='red', label='Demand')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 2: Winter Week')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(SummerWeek.index, SummerWeek['Temp'], label='Temperature')
plt.plot(SummerWeek.index, predictions2_SummerWeek, color='red', label='Demand')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Model 2: Summer Week')
plt.legend()


plt.tight_layout()
plt.show()
