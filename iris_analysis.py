import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Ensures 'plots' directory exists
import os
os.makedirs('plots', exist_ok=True)

# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

# Basic exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Descriptive statistics
print(df.describe())

# Group by species
grouped_means = df.groupby('species').mean()
print(grouped_means)

# Line chart
df.iloc[:, :4].plot(kind='line', title='Feature Trends Across Samples')
plt.xlabel("Sample Index")
plt.ylabel("Feature Values")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('plots/line_chart.png')
plt.show
plt.clf()

# Bar chart
grouped_means['petal length (cm)'].plot(kind='bar', title='Average Petal Length per Species')
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig('plots/bar_chart.png')
plt.show
plt.clf()

# Histogram
df['sepal length (cm)'].plot(kind='hist', bins=20, color='skyblue', title='Distribution of Sepal Length')
plt.xlabel("Sepal Length (cm)")
plt.tight_layout()
plt.savefig('plots/histogram.png')
plt.show
plt.clf()

# Scatter plot
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig('plots/scatter_plot.png')
plt.show
plt.clf()
