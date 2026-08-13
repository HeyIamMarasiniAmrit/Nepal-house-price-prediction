import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

warnings.simplefilter(action="ignore", category=FutureWarning)


plt.rcParams["figure.figsize"] = [10, 5] , full_data = pd.read_csv('/content/titanic.zip'), full_data.shape, (891, 12),full_data.head(), sns.histplot(full_data['Parch'], kde=False), plt.figure(figsize=(8,8))
sns.distplot(full_data['Age'])
plt.show(), sns.relplot(x="Age", y="Fare", col="Pclass", hue="Sex", style="Sex", kind="line", data=full_data)
plt.show()
titanic dataset.csv lai liyara yo sab check gara results succesful aayo aba linkedin , x ani github readme lakhdau