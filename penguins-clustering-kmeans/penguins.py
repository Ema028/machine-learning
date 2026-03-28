import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from utils.pre_processing import *

penguins = sns.load_dataset('penguins')
df = Dataframe(penguins)

df.print_missing() #frações insignificantes
df.drop_missing()
X = df.drop_columns(['species']) #remover alvo

df.print_unique_values() #padronizados
df.pair_plot()
#em alguns gráficos, principalmente os de profundidade do bico, aparecem três agrupamentos, sugere 3 clusters naturais

cat_cols = X.select_dtypes(include=['object', 'str']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=clusters)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.xlabel('bill_length_mm')
plt.ylabel('bill_depth_mm')
plt.title('Clusters com centróides')
plt.show()

plt.scatter(X[:, 0], X[:, 2], c=clusters)
plt.scatter(centroids[:, 2], centroids[:, 3], c='red', marker='X', s=200)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.title('Clusters com centróides')
plt.show()