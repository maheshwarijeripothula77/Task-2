import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)
print(df.head())
print("Shape:", df.shape)
print(df.info())
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
print(df.describe())
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare by Class")
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df['Fare']))
outliers = df[z > 3]

print("Outliers:", len(outliers))
sns.pairplot(df[['Age','Fare','Parents/Children Aboard','Survived']], hue='Survived')
plt.show()
