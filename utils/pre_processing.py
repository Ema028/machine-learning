import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def print_missing(df):
	missing = df.isnull().sum()
	percent = (missing / len(df)) * 100

	missing_table = pd.DataFrame({
		'Missing': missing,
		'Percent (%)': percent
	}).sort_values(by='Missing', ascending=False)
	print(missing_table[missing_table['Missing'] > 0])

def print_unique_values(df):
	for column in df.select_dtypes(include='str').columns:
		print("\n\n")
		print(df[column].value_counts())

def heatmap(df):
	sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
	plt.title("Matriz de correlação para variáveis numéricas")
	plt.show()

def one_hot_heatmap(df):
	#one-hot encoding: cria colunas binárias para cada categoria, evita falsa ordem hierárquica como em label encoding
	cat_cols = df.select_dtypes(include=['object', 'str']).columns
	df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
	df_encoded.head()

	sns.heatmap(df_encoded.corr(), cmap='coolwarm', center=0)
	plt.title("Matriz de correlação com variáveis categóricas")
	plt.show()

def capping_outliers(df, coluna):
	df = df.copy()
	q1 = df[coluna].quantile(0.25)
	q3 = df[coluna].quantile(0.75)
	iqr = q3 - q1

	limite_inferior = q1 - 1.5 * iqr
	limite_superior = q3 + 1.5 * iqr
	df[coluna] = df[coluna].clip(limite_inferior, limite_superior)
	#retorna novo dataframe com outliers limitados pelo IQR
	return df