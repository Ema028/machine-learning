from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.pre_processing import *

df = pd.read_csv("../data/mall.csv", delimiter=',')
data = Dataframe(df.copy())

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

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_normalizado)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)#escala original

idx_Annual_Income = X_normalizado.columns.get_loc('Annual Income (k$)')
idx_Spending_Score = X_normalizado.columns.get_loc('Spending Score (1-100)')

#usar dados não normalizados
plt.scatter(X_original['Annual Income (k$)'], X_original['Spending Score (1-100)'], c=clusters)
plt.scatter(centroids[:, idx_Annual_Income], centroids[:, idx_Spending_Score], c='red', marker='X', s=200)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Clusters com centróides')
plt.show()

df_resultado = df.copy()
df_resultado['Cluster'] = clusters

#agrupa pelo cluster e tira a média de idade, renda e gastos
perfil_numerico = df_resultado.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(1)
print(f"Perfil numérico:\n{perfil_numerico}")

#quantos homens e mulheres caíram em cada grupo
perfil_genero = df_resultado.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
print(f"Perfil de gênero:\n{perfil_genero}")

'''
cluster 0: (Média Renda/Gasto, ~43 anos, +mulheres): foco em custo-benefício
cluster 1: (Alta Renda/Gasto, ~33 anos): foco em marcas premium e fidelidade
cluster 2: (Baixa Renda/Alto Gasto, ~25 anos): foco em tendências e parcelamento
cluster 3: (Alta Renda/Baixo Gasto, ~41 anos): foco em qualidade, exclusividade e restaurantes
cluster 4: (Baixa Renda/Baixo Gasto, ~45 anos): foco em cupons, liquidações e promoções
'''