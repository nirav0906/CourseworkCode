import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dates = pd.date_range(start='2008-1-1', end='2008-7-1')

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
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(df['Temp'], Y, label='Data')
# plt.plot(df['Temp'], results1.fittedvalues, color='red', label='OLS')
# plt.xlabel('Temp')
# plt.ylabel('Demand')
# plt.title('Model 1: Demand vs Temp')
# plt.legend()

# Plot for Model 2
# Since Model 2 is a multiple regression model, we can't plot it on a 2D plot.
# We can plot the demand vs temperature instead.
# plt.subplot(1, 2, 2)
# plt.scatter(df['Temp'], Y, label='Data')
# plt.plot(df['Temp'], results2.fittedvalues, color='red', label='OLS')
# plt.xlabel('Temperature')
# plt.ylabel('Demand')
# plt.title('Model 2: Demand vs Temperature')
# plt.legend()

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
plt.ylabel('Demand (MW)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.title('Model 1: Winter Week')

plt.legend()

plt.subplot(2, 2, 2)
plt.plot(SummerWeek.index, Y_SummerWeek, label='Actual')
plt.plot(SummerWeek.index, predictions1_SummerWeek, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Demand (MW)')
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
plt.ylabel('Demand (MW)')
plt.title('Model 2: Winter Week')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(SummerWeek.index, Y_SummerWeek, label='Actual')
plt.plot(SummerWeek.index, predictions2_SummerWeek, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Demand (MW)')
plt.title('Model 2: Summer Week')
plt.legend()

# plt.tight_layout()
# plt.show()




# Temperature vs Demand

# plt.figure(figsize=(16, 6))
# plt.subplot(2, 2, 1)
# plt.plot(WinterWeek.index, WinterWeek['Temp'], label='Temperature')
# plt.plot(WinterWeek.index, predictions1_WinterWeek, color='red', label='Demand')
# plt.xlabel('Date')
# plt.ylabel('Demand')
# plt.title('Model 1: Winter Week')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(SummerWeek.index, SummerWeek['Temp'], label='Temperature')
# plt.plot(SummerWeek.index, predictions1_SummerWeek, color='red', label='Demand')
# plt.xlabel('Date')
# plt.ylabel('Demand')
# plt.title('Model 1: Summer Week')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(WinterWeek.index, WinterWeek['Temp'], label='Temperature')
# plt.plot(WinterWeek.index, predictions2_WinterWeek, color='red', label='Demand')
# plt.xlabel('Date')
# plt.ylabel('Demand')
# plt.title('Model 2: Winter Week')
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(SummerWeek.index, SummerWeek['Temp'], label='Temperature')
# plt.plot(SummerWeek.index, predictions2_SummerWeek, color='red', label='Demand')
# plt.xlabel('Date')
# plt.ylabel('Demand')
# plt.title('Model 2: Summer Week')
# plt.legend()


# plt.tight_layout()
# plt.show()

# Model 3

hours = df['Hour'].unique()

# Create a dictionary to store the results for each hour
results = {}

for hour in hours:
    # Filter the data for the current hour
    df_hour = df[df['Hour'] == hour]

    # Define the dependent variable
    Y = df_hour['Demand']

    # Define the independent variables
    X = df_hour[['Temp']]
    X = sm.add_constant(X)

    # Run the regression
    model = sm.OLS(Y, X)
    results[hour] = model.fit()

# Create lists to store the values for Beta_0 and Beta_1
beta_0_values = []
beta_1_values = []

# Get the values for Beta_0 and Beta_1 from the results
for hour, result in results.items():
    beta_0_values.append(result.params.const)
    beta_1_values.append(result.params.Temp)

beta_0_values.pop()
beta_1_values.pop()

# Create a list of hours
hours = list(results.keys())
hours.pop()

# Plot Beta_0
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(hours, beta_0_values, marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Beta_0 (Intercept)')
plt.title('Intercept for Each Hour of the Day')
plt.grid(True)

# Plot Beta_1
plt.subplot(1, 2, 2)
plt.plot(hours, beta_1_values, marker='o', color='red')
plt.xlabel('Hour of the Day')
plt.ylabel('Beta_1 (Temperature)')
plt.title('Temperature Coefficient for Each Hour of the Day')
plt.grid(True)

plt.tight_layout()

plt.rcParams.update({'font.size': 6})

plt.figure(figsize=(16, 10))
plt.subplot(1, 2, 1)
df['Time'] = pd.to_datetime(df['Time'])
df = df.set_index('Time')

# Filter dataframe to only include rows at 7 AM
df_7am = df.between_time('07:00', '07:59')
ModeledDemand7 = beta_0_values[6] + beta_1_values[6] * df_7am['Temp']
# Plot demand at 7 AM for each day
plt.plot(df_7am.index, df_7am['Demand'], color='blue')
plt.plot(ModeledDemand7, color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Demand at 7 AM for Each Day')
plt.grid(True)

plt.subplot(1, 2, 2)
# Filter dataframe to only include rows at 23:00
df_23pm = df.between_time('23:00', '23:59')
ModeledDemand23= beta_0_values[22]+beta_1_values[22]*df_23pm['Temp']
# Plot demand at 23:00 for each day
plt.plot(df_23pm.index, df_23pm['Demand'], color='blue')
plt.plot(ModeledDemand23, color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Demand at 23:00 for Each Day')
plt.grid(True)

plt.tight_layout()
plt.show()
