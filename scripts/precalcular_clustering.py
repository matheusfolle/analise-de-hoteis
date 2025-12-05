"""
Script para pré-calcular todos os modelos de clustering
Roda UMA VEZ localmente antes do deploy
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("Iniciando pré-cálculo de clustering...")

# Carrega dados
DATASET_PATH = "../data/raw/data-hoteis-atualizado.xlsx"
if not os.path.exists(DATASET_PATH):
    print(f"ERRO: Dataset não encontrado em {DATASET_PATH}")
    exit(1)

df_bruto = pd.read_excel(DATASET_PATH)
df_filtrado = df_bruto[df_bruto['totalScore'] > 0.2].copy()
print(f"✓ Dataset carregado: {len(df_filtrado)} linhas")

# Prepara features
X = df_filtrado[['totalScore', 'reviewsCount']].copy()
X['reviewsCount_log'] = np.log10(X['reviewsCount'] + 1)
X_for_clustering = X[['totalScore', 'reviewsCount_log']].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_for_clustering)
print("Features preparadas e normalizadas")

# Calcula TODOS os modelos
resultados = {}

print("\n[1/4] Treinando K-Means (k=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)
resultados['K-Means'] = {
    'labels': labels_kmeans,
    'silhouette': silhouette_score(X_scaled, labels_kmeans),
    'davies_bouldin': davies_bouldin_score(X_scaled, labels_kmeans),
    'k': 4
}
print(f"   Silhouette: {resultados['K-Means']['silhouette']:.3f}")

print("\n[2/4] Treinando Hierárquico (k=2)...")
hier = AgglomerativeClustering(n_clusters=2)
labels_hier = hier.fit_predict(X_scaled)
resultados['Hierárquico'] = {
    'labels': labels_hier,
    'silhouette': silhouette_score(X_scaled, labels_hier),
    'davies_bouldin': davies_bouldin_score(X_scaled, labels_hier),
    'k': 2
}
print(f"   Silhouette: {resultados['Hierárquico']['silhouette']:.3f}")

print("\n[3/4] Treinando EM/GMM (k=2)...")
gmm = GaussianMixture(n_components=2, random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)
resultados['EM (GMM)'] = {
    'labels': labels_gmm,
    'silhouette': silhouette_score(X_scaled, labels_gmm),
    'davies_bouldin': davies_bouldin_score(X_scaled, labels_gmm),
    'k': 2
}
print(f"   Silhouette: {resultados['EM (GMM)']['silhouette']:.3f}")

print("\n[4/4] Treinando DBSCAN (eps=0.7, min_samples=5)...")
dbscan = DBSCAN(eps=0.7, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

if n_clusters >= 2:
    mask = labels_dbscan != -1
    sil = silhouette_score(X_scaled[mask], labels_dbscan[mask])
    dav = davies_bouldin_score(X_scaled[mask], labels_dbscan[mask])
else:
    sil = -1.0
    dav = 999.0

resultados['DBSCAN'] = {
    'labels': labels_dbscan,
    'silhouette': sil,
    'davies_bouldin': dav,
    'k': None
}
print(f"   Clusters encontrados: {n_clusters}")
print(f"   Silhouette: {sil:.3f}" if sil != -1.0 else "   Silhouette: N/A")

# Prepara DataFrame base para plots
df_plot_base = X[['totalScore', 'reviewsCount']].copy()

# Cria diretório se não existir
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)

# Salva tudo
output_path = os.path.join(output_dir, "clustering_results.pkl")
with open(output_path, 'wb') as f:
    pickle.dump({
        'resultados': resultados,
        'df_plot': df_plot_base
    }, f)

print(f"\n{'='*60}")
print(f"✓ SUCESSO: Clustering pré-calculado salvo em:")
print(f"  {output_path}")
print(f"{'='*60}")
print("\nAgora eu posso fazer deploy sem medo de 502!")