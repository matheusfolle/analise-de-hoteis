"""
ANÁLISE ESTATÍSTICA COMPLETA - VERSÃO FINAL
Calcula TODAS as métricas para Entrega 1
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CARREGAR DATASET
# ============================================

url = 'https://gist.githubusercontent.com/matheusfolle/b829cc7cbd10a5cbf9888a901f8b691e/raw/55b4552f4b91d6b727b868d4e8670fc976e96680/data-hoteis-atualizado'

print("📡 Carregando dataset...")
df = pd.read_csv(url, sep=';')

# Criar has_website
df['has_website'] = df['website'].notna() & (df['website'] != '')

# Separar grupos
df_com = df[df['has_website'] == True].copy()
df_sem = df[df['has_website'] == False].copy()

print(f"✓ Total: {len(df)} | COM: {len(df_com)} | SEM: {len(df_sem)}\n")

# ============================================
# FUNÇÃO PARA CALCULAR TODAS AS MÉTRICAS
# ============================================

def calcular_metricas(data, col_name):
    """Calcula todas as métricas estatísticas"""
    serie = data[col_name].dropna()

    metricas = {
        'Média': serie.mean(),
        'Mediana': serie.median(),
        'Moda': serie.mode()[0] if len(serie.mode()) > 0 else np.nan,
        'Desvio Padrão': serie.std(),
        'Variância': serie.var(),
        'Mínimo': serie.min(),
        'Máximo': serie.max(),
        'Contagem': len(serie),
        'Q1 (25%)': serie.quantile(0.25),
        'Q2 (50%)': serie.quantile(0.50),  # = Mediana
        'Q3 (75%)': serie.quantile(0.75),
        'Amplitude': serie.max() - serie.min(),
        'Curtose': serie.kurtosis(),
        'Assimetria': serie.skew(),
    }

    # Coeficiente de Variação
    if metricas['Média'] != 0:
        metricas['CV (%)'] = (metricas['Desvio Padrão'] / metricas['Média']) * 100
    else:
        metricas['CV (%)'] = np.nan

    return metricas


# ============================================
# ANÁLISE DO DATASET COMPLETO (GERAL)
# ============================================

print("="*70)
print("DATASET COMPLETO - ESTATÍSTICAS GERAIS")
print("="*70)

# Para reviewsCount
print("\n--- Análise Geral: reviewsCount ---")
metricas_rev_geral = calcular_metricas(df, 'reviewsCount')
tabela_rev_geral = pd.DataFrame({
    'Métrica': metricas_rev_geral.keys(),
    'Valor': metricas_rev_geral.values()
})
print(tabela_rev_geral.to_string(index=False))

# Para totalScore
print("\n--- Análise Geral: totalScore ---")
metricas_score_geral = calcular_metricas(df, 'totalScore')
tabela_score_geral = pd.DataFrame({
    'Métrica': metricas_score_geral.keys(),
    'Valor': metricas_score_geral.values()
})
print(tabela_score_geral.to_string(index=False))


# ============================================
# CALCULAR PARA reviewsCount
# ============================================

print("="*70)
print("REVIEWSCOUNT - ESTATÍSTICAS COMPLETAS")
print("="*70)

metricas_rev_com = calcular_metricas(df_com, 'reviewsCount')
metricas_rev_sem = calcular_metricas(df_sem, 'reviewsCount')

tabela_reviews = pd.DataFrame({
    'Métrica': metricas_rev_com.keys(),
    'COM Website': metricas_rev_com.values(),
    'SEM Website': metricas_rev_sem.values()
})

# Diferença %
tabela_reviews['Diferença (%)'] = ((tabela_reviews['COM Website'] - tabela_reviews['SEM Website']) /
                                     tabela_reviews['SEM Website'] * 100).round(2)

print(tabela_reviews.to_string(index=False))

# ============================================
# CALCULAR PARA totalScore
# ============================================

print("\n" + "="*70)
print("TOTALSCORE - ESTATÍSTICAS COMPLETAS")
print("="*70)

metricas_score_com = calcular_metricas(df_com, 'totalScore')
metricas_score_sem = calcular_metricas(df_sem, 'totalScore')

tabela_score = pd.DataFrame({
    'Métrica': metricas_score_com.keys(),
    'COM Website': metricas_score_com.values(),
    'SEM Website': metricas_score_sem.values()
})

tabela_score['Diferença (%)'] = ((tabela_score['COM Website'] - tabela_score['SEM Website']) /
                                   tabela_score['SEM Website'] * 100).round(2)

print(tabela_score.to_string(index=False))

# ============================================
# COVARIÂNCIA
# ============================================

print("\n" + "="*70)
print("COVARIÂNCIA: totalScore vs reviewsCount")
print("="*70)

cov_com = df_com[['totalScore', 'reviewsCount']].cov().iloc[0, 1]
cov_sem = df_sem[['totalScore', 'reviewsCount']].cov().iloc[0, 1]

print(f"COM Website: {cov_com:.2f}")
print(f"SEM Website: {cov_sem:.2f}")

# ============================================
# CORRELAÇÃO
# ============================================

print("\n" + "="*70)
print("CORRELAÇÃO (Pearson): totalScore vs reviewsCount")
print("="*70)

corr_com = df_com[['totalScore', 'reviewsCount']].corr().iloc[0, 1]
corr_sem = df_sem[['totalScore', 'reviewsCount']].corr().iloc[0, 1]

print(f"COM Website: {corr_com:.4f}")
print(f"SEM Website: {corr_sem:.4f}")

# ============================================
# TESTE T-STUDENT
# ============================================

print("\n" + "="*70)
print("TESTE T-STUDENT")
print("="*70)

t_score, p_score = stats.ttest_ind(df_com['totalScore'].dropna(), df_sem['totalScore'].dropna())
t_reviews, p_reviews = stats.ttest_ind(df_com['reviewsCount'].dropna(), df_sem['reviewsCount'].dropna())

print(f"totalScore:")
print(f"  t-statistic: {t_score:.4f}")
print(f"  p-value: {p_score:.6f}")
print(f"  Significativo: {'SIM ✓' if p_score < 0.05 else 'NÃO ✗'}")

print(f"\nreviewsCount:")
print(f"  t-statistic: {t_reviews:.4f}")
print(f"  p-value: {p_reviews:.6f}")
print(f"  Significativo: {'SIM ✓' if p_reviews < 0.05 else 'NÃO ✗'}")

# ============================================
# TABELA RESUMO CONSOLIDADA
# ============================================

print("\n" + "="*70)
print("TABELA RESUMO - VALORES PRINCIPAIS")
print("="*70)

resumo = pd.DataFrame({
    'Métrica': ['Média', 'Mediana', 'Desvio Padrão', 'Q1', 'Q3', 'Curtose', 'Assimetria', 'CV (%)'],
    'Reviews COM': [
        metricas_rev_com['Média'],
        metricas_rev_com['Mediana'],
        metricas_rev_com['Desvio Padrão'],
        metricas_rev_com['Q1 (25%)'],
        metricas_rev_com['Q3 (75%)'],
        metricas_rev_com['Curtose'],
        metricas_rev_com['Assimetria'],
        metricas_rev_com['CV (%)']
    ],
    'Reviews SEM': [
        metricas_rev_sem['Média'],
        metricas_rev_sem['Mediana'],
        metricas_rev_sem['Desvio Padrão'],
        metricas_rev_sem['Q1 (25%)'],
        metricas_rev_sem['Q3 (75%)'],
        metricas_rev_sem['Curtose'],
        metricas_rev_sem['Assimetria'],
        metricas_rev_sem['CV (%)']
    ],
    'Score COM': [
        metricas_score_com['Média'],
        metricas_score_com['Mediana'],
        metricas_score_com['Desvio Padrão'],
        metricas_score_com['Q1 (25%)'],
        metricas_score_com['Q3 (75%)'],
        metricas_score_com['Curtose'],
        metricas_score_com['Assimetria'],
        metricas_score_com['CV (%)']
    ],
    'Score SEM': [
        metricas_score_sem['Média'],
        metricas_score_sem['Mediana'],
        metricas_score_sem['Desvio Padrão'],
        metricas_score_sem['Q1 (25%)'],
        metricas_score_sem['Q3 (75%)'],
        metricas_score_sem['Curtose'],
        metricas_score_sem['Assimetria'],
        metricas_score_sem['CV (%)']
    ]
})

print(resumo.to_string(index=False))

# ============================================
# FIM DO SCRIPT
# ============================================