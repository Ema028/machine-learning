from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from utils.pre_processing import *

scaler = StandardScaler()

def main():
    penguins = sns.load_dataset('penguins')
    df = Dataframe(penguins)

    df.print_missing()  # frações insignificantes
    df.drop_missing()
    X_original = df.drop_columns(['species'])  # remover alvo

    df.print_unique_values()  # padronizados
    df.pair_plot()
    # em alguns gráficos, principalmente os de profundidade do bico, aparecem três agrupamentos, sugere 3 clusters naturais

    X = X_original.copy()
    cat_cols = X.select_dtypes(include=['object', 'str']).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = scaler.fit_transform(X)
    inertia = []
    silhouette_scores = []
    K = range(2, 10)

    for k in K:
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        labels = kmeans_temp.fit_predict(X)
        inertia.append(kmeans_temp.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    plt.figure()
    plt.plot(K, inertia, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inertia')
    plt.title('Método de Elbow')
    plt.show()
    #queda muito forte de 2 pra 4, depois achata

    plt.figure()
    plt.plot(K, silhouette_scores, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    plt.show()
    #crescimento até 4, estabilização até 7(aumentar k não melhora muito a separação real)
    # volta a crescer em 8 provavelmente por over-segmentation

    #comparar k=3 pela análise exploratória inicial e k=4 pelo equilíbrio entre menor inércio, maior silhouette
    kmeans(3, X, X_original, penguins)
    kmeans(4, X, X_original, penguins)
    #modelo com k=4 apresentou melhor desempenho mesmo com só 3 espécies, provavelmente subestruturas dentro das espécies

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