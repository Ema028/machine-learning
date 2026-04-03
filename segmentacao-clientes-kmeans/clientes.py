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
#'CustomerID' removido por ser só identificador,
#'Gender' porque distância espacial entre 0 e 1 estava criando um ruído artificial nos grupos de consumo
#'Age' porque estava desfocando do comportamento financeiro, faz sentido cruzar dados no final
X_original = data.drop_columns(['CustomerID', 'Gender', 'Age'])
X_normalizado = pd.DataFrame(scaler.fit_transform(data.df), columns=data.df.columns)

inertia = []
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels = kmeans_temp.fit_predict(X_normalizado)
    inertia.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_normalizado, labels))

plt.figure()
plt.plot(K, inertia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inertia')
plt.title('Método de Elbow')
plt.show()
#cotovelo claro em 5

plt.figure()
plt.plot(K, silhouette_scores, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.show()
#pico claro em 5