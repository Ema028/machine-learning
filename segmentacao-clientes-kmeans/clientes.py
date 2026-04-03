from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from utils.pre_processing import *

df = pd.read_csv("../data/mall.csv", delimiter=',')
data = Dataframe(df)

print(df.info())
data.print_missing() #nada faltando
data.print_unique_values() #já padronizado
print(df.describe().T) #sinais de outliers em Annual Income

data.histogram('Annual Income (k$)', "Distribuição de Renda Anual")
data.box_plot(None, 'Annual Income (k$)', "Boxplot de Renda Anual")
#cauda longa no histograma, mas só 1 valor fora do intervalo interquartil no boxplot

data.capping_outliers(['Annual Income (k$)'])
data.pair_plot()
#pelo gráfico o id está ordenado por renda, o grafico annual income x spending score indica 5 clusters naturais

scaler = StandardScaler()
X_original = data.drop_columns(['CustomerID'])
X_normalizado = pd.DataFrame(scaler.fit_transform(data.df), columns=data.df.columns)