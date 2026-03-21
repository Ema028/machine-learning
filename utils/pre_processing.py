import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class Dataframe:
	def __init__(self, df):
		self.df = df
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None

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

	#dividir base em conjuntos de treino (por default 80%) e teste (por default 20%)
	def separar_base(self, target_column, test_size=0.2, random_state=42):
		x = self.df.drop(columns=[target_column]) #x é todas as variáveis menos target
		y = self.df[target_column] #y é target

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			x, y, test_size=test_size, random_state=random_state
		)