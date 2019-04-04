import pandas as pd
from statsmodels.formula.api import ols
from sklearn.datasets import load_boston

boston = load_boston() # Boston Data set is a dictionary

print(boston.keys()) # checking the keys that are part of the dictionary
print(boston.data.shape) # There are 506 rows of data with a total of 13 columns
print(boston.feature_names)

print(boston.DESCR) # Exploring the data set

# summary of the different columns involved using the pandas package
BostonPanda = pd.DataFrame(boston.data)
BostonPanda.columns = boston.feature_names
print(BostonPanda.head()) # does not include boston house price?

BostonPanda['PRICE'] = boston.target
print(BostonPanda.head())

# Now we can finally perform summary statistics
print(BostonPanda.describe())

# Correlation
print(BostonPanda.corr())

model = ols("PRICE ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT", data = BostonPanda)
results = model.fit()
print(results.summary())

# The P Value of the individual explanatory variables will show which of the
# Explanatory variables has a higher or lower statistical significance or influence
# on the response variable which in this case is Price.
# The P value is defined as the probability under the assumption of no effect or no difference
# The P stands for probability and measures how likely it is that any observed difference between groups is due to chance
# the explanatory variable with the least significance/ influence is AGE as it has the highest P Value
# The explanatory variableS with the highest significance/ influence are NOX, RM DIS, RAD, PTRatio and LSTAT
# These all have a p value of pretty much zero indicating that they are significant predictor variables