import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Dataframe:
	def __init__(self, df):
		self.df = df

	def print_missing(self):
		missing = self.df.isnull().sum()
		percent = (missing / len(self.df)) * 100

		missing_table = pd.DataFrame({
			'Missing': missing,
			'Percent (%)': percent
		}).sort_values(by='Missing', ascending=False)
		print(missing_table[missing_table['Missing'] > 0])

	def print_unique_values(self):
		for column in self.df.select_dtypes(include='str').columns:
			print("\n\n")
			print(self.df[column].value_counts())

	def heatmap(self):
		sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
		plt.title("Matriz de correlação para variáveis numéricas")
		plt.show()

	def one_hot_heatmap(self):
		#one-hot encoding: cria colunas binárias para cada categoria, evita falsa ordem hierárquica como em label encoding
		cat_cols = self.df.select_dtypes(include=['object', 'str']).columns
		df_encoded = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
		df_encoded.head()

		sns.heatmap(df_encoded.corr(), cmap='coolwarm', center=0)
		plt.title("Matriz de correlação com variáveis categóricas")
		plt.show()

	def capping_outliers(self, coluna):
		df = self.df.copy()
		q1 = df[coluna].quantile(0.25)
		q3 = df[coluna].quantile(0.75)
		iqr = q3 - q1

		limite_inferior = q1 - 1.5 * iqr
		limite_superior = q3 + 1.5 * iqr
		df[coluna] = df[coluna].clip(limite_inferior, limite_superior)
		#retorna novo dataframe com outliers limitados pelo IQR
		return self.df

	def apply_log(self, columns):
		for col in columns:
			self.df[col] = np.log1p(self.df[col]) #log +1 para caso de 0