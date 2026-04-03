from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from utils.pre_processing import *

scaler = StandardScaler()

def main():
    penguins = sns.load_dataset('penguins')
    df = Dataframe(penguins)

    df.print_missing()  # frações insignificantes
    df.drop_missing()
    X_original = df.drop_columns(['species', 'island', 'sex'])  #remover alvo e categóricas

    df.print_unique_values()  # padronizados
    df.pair_plot()
    # em alguns gráficos, principalmente os de profundidade do bico, aparecem três agrupamentos, sugere 3 clusters naturais

    X_normalizado = pd.DataFrame(scaler.fit_transform(df.df), columns=df.df.columns)
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
    #queda brusca de 2 pra 3, depois continua a cair em taxas menores

    plt.figure()
    plt.plot(K, silhouette_scores, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.show()
    #pico em k=2, provavelmente porque os gentoo são muito grandes, cria uma divisão matemática

    #comparar k=3 pela análise exploratória inicial, k=2 pelo silhouette e k=4
    kmeans(3, X_normalizado, X_original, penguins)
    kmeans(2, X_normalizado, X_original, penguins)
    kmeans(4, X_normalizado, X_original, penguins)
    #modelo com k=3 apresentou melhor desempenho

def kmeans(k, X, X_original, penguins):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)#escala original

    idx_bill_length = X.columns.get_loc('bill_length_mm')
    idx_bill_depth = X.columns.get_loc('bill_depth_mm')
    idx_flipper = X.columns.get_loc('flipper_length_mm')

    #usar dados não normalizados
    plt.scatter(X_original['bill_length_mm'], X_original['bill_depth_mm'], c=clusters)
    plt.scatter(centroids[:, idx_bill_length], centroids[:, idx_bill_depth], c='red', marker='X', s=200)
    plt.xlabel('bill_length_mm')
    plt.ylabel('bill_depth_mm')
    plt.title('Clusters com centróides')
    plt.show()

    plt.scatter(X_original['bill_length_mm'], X_original['flipper_length_mm'], c=clusters)
    plt.scatter(centroids[:, idx_bill_length], centroids[:, idx_flipper], c='red', marker='X', s=200)
    plt.xlabel('bill_length_mm')
    plt.ylabel('flipper_length_mm')
    plt.title('Clusters com centróides')
    plt.show()

    #comparação clusters com as espécies reais
    true_labels = penguins['species'].astype('category').cat.codes
    true_labels = true_labels[X_original.index] #alinhar com dados limpos, X_original tem menos linhas
    print(f"Adjusted Rand Index(k={k}): {adjusted_rand_score(true_labels, clusters)}")

main()