### ANÁLISE ESTATÍSTICA COMPARATIVA: HOTÉIS COM VS. SEM WEBSITE
#
# Este script realiza uma análise estatística descritiva e de teste de hipóteses
# para comparar a performance (totalScore) e o engajamento (reviewsCount)
# de hotéis com base na presença (ou ausência) de um website.
#
# OBJETIVO: Validar a hipótese de que hotéis com website apresentam
# melhores indicadores.
#
# Autor: Seu Nome / Gemini
# Data: 06/11/2025
#
# Bibliotecas necessárias: pandas, numpy, scipy, openpyxl
# (matplotlib e seaborn são importados, mas não usados ativamente neste script)
###

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os # Usado para verificar a saída do Excel

# --- Configurações Iniciais ---
EXCEL_OUTPUT_FILE = 'estatisticas_descritivas.xlsx'
REQUIRED_COLS = ['totalScore', 'reviewsCount', 'website']
MIN_SAMPLE_SIZE = 30 # Mínimo para o Teste T ser mais confiável

# Variáveis globais para guardar resultados e usar na interpretação
p_value_score = 1.0
p_value_reviews = 1.0
p_value_corr = 1.0
corr_com_site = 0.0
corr_sem_site = 0.0
stats_totalscore = pd.DataFrame()
stats_reviews = pd.DataFrame()
df_com_site = pd.DataFrame()
df_sem_site = pd.DataFrame()


### PARTE 1: PREPARAÇÃO DOS DADOS
print("--- PARTE 1: PREPARAÇÃO DOS DADOS ---")

try:
    # Column names (in order)
    column_names = ['title', 'totalScore', 'reviewsCount', 'street', 'city',
                    'state', 'countryCode', 'website', 'phone', 'categoryName', 'url']

    # Load CSV without header, then assign column names
    url = 'https://gist.githubusercontent.com/matheusfolle/22f2108ec86a4cd118f0c26dec4e6e8c/raw/d862b22bab3b520164e410c877d49d2b570cb07d/data-hoteis.csv'

    df = pd.read_csv(url, header=None, names=column_names, sep='\t', engine='python', on_bad_lines='skip')
    print("✓ Dataset loaded successfully!")


except Exception as e:
    print(f"ERRO ao carregar o arquivo: {e}")
    sys.exit(1)

# Stop Condition 2: Verificar colunas
missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
if missing_cols:
    print(f"ERRO: Coluna(s) {missing_cols} não encontrada(s) no dataset.")
    print(f"Colunas disponíveis: {list(df.columns)}")
    sys.exit(1)

# Limpar dados: Converter colunas para numérico, forçando erros para NaN
df['totalScore'] = pd.to_numeric(df['totalScore'], errors='coerce')
df['reviewsCount'] = pd.to_numeric(df['reviewsCount'], errors='coerce')

# Remover linhas onde as métricas principais são nulas
df.dropna(subset=['totalScore', 'reviewsCount'], inplace=True)

# Criar coluna derivada has_website
# .fillna('') garante que valores nulos sejam tratados como string vazia
df['has_website'] = df['website'].fillna('').str.strip().notna() & (df['website'].fillna('').str.strip() != '')

# Separar em dois dataframes
df_com_site = df[df['has_website'] == True].copy()
df_sem_site = df[df['has_website'] == False].copy()

# Verificar distribuição
print(f"Total de hotéis (após limpeza): {len(df)}")
print(f"Hotéis COM website: {len(df_com_site)} ({len(df_com_site)/len(df)*100:.1f}%)")
print(f"Hotéis SEM website: {len(df_sem_site)} ({len(df_sem_site)/len(df)*100:.1f}%)")

# Stop Condition 3: Verificar tamanho da amostra
if len(df_com_site) < MIN_SAMPLE_SIZE:
    print(f"AVISO: Amostra pequena no grupo [COM Website] (N={len(df_com_site)} < {MIN_SAMPLE_SIZE}). Resultados podem não ser confiáveis.")
if len(df_sem_site) < MIN_SAMPLE_SIZE:
    print(f"AVISO: Amostra pequena no grupo [SEM Website] (N={len(df_sem_site)} < {MIN_SAMPLE_SIZE}). Resultados podem não ser confiáveis.")

# Verificar se algum grupo está vazio
if df_com_site.empty or df_sem_site.empty:
    print("ERRO: Um dos grupos (COM ou SEM website) está vazio. Impossível continuar a análise comparativa.")
    sys.exit(1)


### PARTE 2: ESTATÍSTICAS DESCRITIVAS - TOTALSCORE
try:
    stats_totalscore = pd.DataFrame({
        'Métrica': ['Média', 'Mediana', 'Moda', 'Desvio Padrão', 'Variância', 'Mínimo', 'Máximo', 'Q1 (25%)', 'Q3 (75%)'],
        'COM Website': [
            df_com_site['totalScore'].mean(),
            df_com_site['totalScore'].median(),
            df_com_site['totalScore'].mode()[0] if len(df_com_site['totalScore'].mode()) > 0 else np.nan,
            df_com_site['totalScore'].std(),
            df_com_site['totalScore'].var(),
            df_com_site['totalScore'].min(),
            df_com_site['totalScore'].max(),
            df_com_site['totalScore'].quantile(0.25),
            df_com_site['totalScore'].quantile(0.75)
        ],
        'SEM Website': [
            df_sem_site['totalScore'].mean(),
            df_sem_site['totalScore'].median(),
            df_sem_site['totalScore'].mode()[0] if len(df_sem_site['totalScore'].mode()) > 0 else np.nan,
            df_sem_site['totalScore'].std(),
            df_sem_site['totalScore'].var(),
            df_sem_site['totalScore'].min(),
            df_sem_site['totalScore'].max(),
            df_sem_site['totalScore'].quantile(0.25),
            df_sem_site['totalScore'].quantile(0.75)
        ]
    })

    # Stop Condition 6: Garantir formatação com 2 casas decimais
    stats_totalscore['COM Website'] = stats_totalscore['COM Website'].round(2)
    stats_totalscore['SEM Website'] = stats_totalscore['SEM Website'].round(2)

    # Calcular diferença percentual (evitar divisão por zero)
    if stats_totalscore.loc[stats_totalscore['Métrica'] == 'Média', 'SEM Website'].values[0] != 0:
        stats_totalscore['Diferença (%)'] = ((stats_totalscore['COM Website'] - stats_totalscore['SEM Website']) / stats_totalscore['SEM Website'] * 100).round(2)
    else:
        stats_totalscore['Diferença (%)'] = np.nan

    print("\n=== ESTATÍSTICAS DESCRITIVAS: TOTALSCORE ===")
    print(stats_totalscore.to_string(index=False))

except Exception as e:
    print(f"\nERRO ao calcular estatísticas de TotalScore: {e}")


### PARTE 3: ESTATÍSTICAS DESCRITIVAS - REVIEWSCOUNT
try:
    stats_reviews = pd.DataFrame({
        'Métrica': ['Média', 'Mediana', 'Moda', 'Desvio Padrão', 'Variância', 'Mínimo', 'Máximo', 'Q1 (25%)', 'Q3 (75%)'],
        'COM Website': [
            df_com_site['reviewsCount'].mean(),
            df_com_site['reviewsCount'].median(),
            df_com_site['reviewsCount'].mode()[0] if len(df_com_site['reviewsCount'].mode()) > 0 else np.nan,
            df_com_site['reviewsCount'].std(),
            df_com_site['reviewsCount'].var(),
            df_com_site['reviewsCount'].min(),
            df_com_site['reviewsCount'].max(),
            df_com_site['reviewsCount'].quantile(0.25),
            df_com_site['reviewsCount'].quantile(0.75)
        ],
        'SEM Website': [
            df_sem_site['reviewsCount'].mean(),
            df_sem_site['reviewsCount'].median(),
            df_sem_site['reviewsCount'].mode()[0] if len(df_sem_site['reviewsCount'].mode()) > 0 else np.nan,
            df_sem_site['reviewsCount'].std(),
            df_sem_site['reviewsCount'].var(),
            df_sem_site['reviewsCount'].min(),
            df_sem_site['reviewsCount'].max(),
            df_sem_site['reviewsCount'].quantile(0.25),
            df_sem_site['reviewsCount'].quantile(0.75)
        ]
    })

    # Stop Condition 6: Garantir formatação com 2 casas decimais
    stats_reviews['COM Website'] = stats_reviews['COM Website'].round(2)
    stats_reviews['SEM Website'] = stats_reviews['SEM Website'].round(2)

    # Calcular diferença percentual (evitar divisão por zero)
    if stats_reviews.loc[stats_reviews['Métrica'] == 'Média', 'SEM Website'].values[0] != 0:
        stats_reviews['Diferença (%)'] = ((stats_reviews['COM Website'] - stats_reviews['SEM Website']) / stats_reviews['SEM Website'] * 100).round(2)
    else:
        stats_reviews['Diferença (%)'] = np.nan


    print("\n=== ESTATÍSTICAS DESCRITIVAS: REVIEWSCOUNT ===")
    print(stats_reviews.to_string(index=False))

except Exception as e:
    print(f"\nERRO ao calcular estatísticas de ReviewsCount: {e}")


### PARTE 4: TESTE T-STUDENT (SIGNIFICÂNCIA ESTATÍSTICA)
print("\n=== TESTE T-STUDENT (Significância Estatística) ===")
try:
    # Teste T para totalScore
    # Usar equal_var=False (Teste de Welch) é mais robusto quando as variâncias são diferentes
    # (o que é provável, vide Parte 2 e 3)
    t_stat_score, p_value_score = stats.ttest_ind(
        df_com_site['totalScore'].dropna(),
        df_sem_site['totalScore'].dropna(),
        equal_var=False
    )

    # Teste T para reviewsCount
    t_stat_reviews, p_value_reviews = stats.ttest_ind(
        df_com_site['reviewsCount'].dropna(),
        df_sem_site['reviewsCount'].dropna(),
        equal_var=False
    )

    print(f"TotalScore:")
    print(f" t-statistic: {t_stat_score:.4f}")
    print(f" p-value: {p_value_score: .6f}") # Usar .e para valores muito pequenos
    print(f" Diferença significativa: {'SIM ✓' if p_value_score < 0.05 else 'NÃO ✗'}")

    print(f"\nReviewsCount:")
    print(f" t-statistic: {t_stat_reviews:.4f}")
    print(f" p-value: {p_value_reviews: .6f}")
    print(f" Diferença significativa: {'SIM ✓' if p_value_reviews < 0.05 else 'NÃO ✗'}")

except Exception as e:
    print(f"ERRO ao executar o Teste T: {e}")
    print("Os p-values serão definidos como 1.0 para continuar a interpretação.")
    p_value_score = 1.0
    p_value_reviews = 1.0


### PARTE 5: CORRELAÇÃO (PEARSON)
print("\n=== CORRELAÇÃO (Pearson): totalScore vs reviewsCount ===")
try:
    # Correlação entre totalScore e reviewsCount para cada grupo
    corr_com_site = df_com_site[['totalScore', 'reviewsCount']].corr().iloc[0, 1]
    corr_sem_site = df_sem_site[['totalScore', 'reviewsCount']].corr().iloc[0, 1]

    print(f"Hotéis COM website: {corr_com_site:.4f}")
    print(f"Hotéis SEM website: {corr_sem_site:.4f}")
    print(f"Diferença: {abs(corr_com_site - corr_sem_site):.4f}")

    # Testar significância da diferença entre correlações (Fisher Z-transform)
    n_com = len(df_com_site)
    n_sem = len(df_sem_site)

    # Evitar log(0) se correlação for 1 ou -1
    z_com = 0.5 * np.log((1 + corr_com_site + 1e-9) / (1 - corr_com_site + 1e-9))
    z_sem = 0.5 * np.log((1 + corr_sem_site + 1e-9) / (1 - corr_sem_site + 1e-9))

    z_diff = (z_com - z_sem) / np.sqrt(1/(n_com-3) + 1/(n_sem-3))
    p_value_corr = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    print(f"p-value (Fisher Z-transform): {p_value_corr:.6f}")
    print(f"Correlações são estatisticamente diferentes: {'SIM ✓' if p_value_corr < 0.05 else 'NÃO ✗'}")

except Exception as e:
    print(f"ERRO ao calcular correlação: {e}")
    corr_com_site = 0.0
    corr_sem_site = 0.0
    p_value_corr = 1.0


### PARTE 6: COVARIÂNCIA
print("\n=== COVARIÂNCIA: totalScore vs reviewsCount ===")
try:
    # Calcular covariância
    cov_com_site = df_com_site[['totalScore', 'reviewsCount']].cov().iloc[0, 1]
    cov_sem_site = df_sem_site[['totalScore', 'reviewsCount']].cov().iloc[0, 1]

    print(f"Hotéis COM website: {cov_com_site:.2f}")
    print(f"Hotéis SEM website: {cov_sem_site:.2f}")

except Exception as e:
    print(f"ERRO ao calcular covariância: {e}")


### PARTE 7: INTERPRETAÇÃO CIENTÍFICA
print("\n" + "="*70)
print("INTERPRETAÇÃO CIENTÍFICA DOS RESULTADOS")
print("="*70)

try:
    # Calcular diferenças percentuais para interpretação
    mean_score_com = df_com_site['totalScore'].mean()
    mean_score_sem = df_sem_site['totalScore'].mean()
    mean_reviews_com = df_com_site['reviewsCount'].mean()
    mean_reviews_sem = df_sem_site['reviewsCount'].mean()

    diff_score = ((mean_score_com - mean_score_sem) / mean_score_sem * 100) if mean_score_sem != 0 else 0
    diff_reviews = ((mean_reviews_com - mean_reviews_sem) / mean_reviews_sem * 100) if mean_reviews_sem != 0 else 0

    def get_corr_strength(r):
        r_abs = abs(r)
        if r_abs > 0.7: return 'forte'
        if r_abs > 0.4: return 'moderada'
        if r_abs > 0.2: return 'fraca'
        return 'muito fraca'

    interpretacao = f"""
1. DESEMPENHO (totalScore):
- Hotéis COM website: média de {mean_score_com:.2f}
- Hotéis SEM website: média de {mean_score_sem:.2f}
- Diferença: {diff_score:+.1f}% (COM Website vs. SEM Website)
- Significância estatística: {'CONFIRMADA (p < 0.05)' if p_value_score < 0.05 else 'NÃO CONFIRMADA (p >= 0.05)'}

2. ENGAJAMENTO (reviewsCount):
- Hotéis COM website: média de {mean_reviews_com:.0f} avaliações
- Hotéis SEM website: média de {mean_reviews_sem:.0f} avaliações
- Diferença: {diff_reviews:+.1f}% (COM Website vs. SEM Website)
- Significância estatística: {'CONFIRMADA (p < 0.05)' if p_value_reviews < 0.05 else 'NÃO CONFIRMADA (p >= 0.05)'}

3. CORRELAÇÃO (totalScore ↔ reviewsCount):
- Hotéis COM website: r = {corr_com_site:.3f} (correlação {get_corr_strength(corr_com_site)} positiva)
- Hotéis SEM website: r = {corr_sem_site:.3f} (correlação {get_corr_strength(corr_sem_site)} positiva)
- A relação entre avaliações e desempenho é {'MAIS FORTE' if corr_com_site > corr_sem_site else 'MAIS FRACA'} em hotéis COM website.
- Esta diferença nas correlações {'É' if p_value_corr < 0.05 else 'NÃO É'} estatisticamente significativa.

4. CONCLUSÃO PRELIMINAR:
{'Os dados fornecem evidências estatisticamente significativas (p < 0.05) de que a presença de website está associada tanto a uma melhor performance (notas maiores) quanto a um maior engajamento (mais avaliações). A hipótese inicial do projeto é validada por esta análise descritiva.' if (p_value_score < 0.05 and p_value_reviews < 0.05) else 'Os dados NÃO fornecem evidências estatísticas suficientes para validar a hipótese inicial. Embora possam existir diferenças, elas não são estatisticamente significativas (p >= 0.05) para uma ou ambas as métricas.'}
"""

    print(interpretacao)

    # --- Verificações Adicionais (Stop Conditions) ---

    # Stop Condition 4: p > 0.05
    if p_value_score >= 0.05 and p_value_reviews >= 0.05:
        print("\nATENÇÃO: Diferenças NÃO são estatisticamente significativas (p >= 0.05) para NENHUMA das métricas.")
        print(" Considere análises adicionais ou coletas de mais dados.")
    elif p_value_score >= 0.05:
        print("\nATENÇÃO: A diferença em totalScore NÃO é estatisticamente significativa (p >= 0.05).")
    elif p_value_reviews >= 0.05:
        print("\nATENÇÃO: A diferença em reviewsCount NÃO é estatisticamente significativa (p >= 0.05).")

    # Stop Condition 5: Correlação fraca/negativa
    if corr_com_site < 0.2 and corr_sem_site < 0.2:
        print(f"\nNOTA: Correlação muito fraca (|r| < 0.2) detectada em AMBOS os grupos.")
        print(" Revisar se 'totalScore' e 'reviewsCount' são as melhores métricas de associação.")
    elif corr_com_site < 0 or corr_sem_site < 0:
        print(f"\nNOTA: Correlação NEGATIVA detectada. Revisar dados.")

except Exception as e:
    print(f"ERRO ao gerar interpretação: {e}")


### PARTE 8: EXPORTAR TABELAS PARA EXCEL
try:
    # Criar arquivo Excel com múltiplas abas
    with pd.ExcelWriter(EXCEL_OUTPUT_FILE, engine='openpyxl') as writer:

        # Usar as tabelas já formatadas e arredondadas
        if not stats_totalscore.empty:
            stats_totalscore.to_excel(writer, sheet_name='TotalScore', index=False)

        if not stats_reviews.empty:
            stats_reviews.to_excel(writer, sheet_name='ReviewsCount', index=False)

        # Criar aba de resumo
        resumo = pd.DataFrame({
            'Grupo': ['COM Website', 'SEM Website'],
            'Média totalScore': [df_com_site['totalScore'].mean(), df_sem_site['totalScore'].mean()],
            'Média reviewsCount': [df_com_site['reviewsCount'].mean(), df_sem_site['reviewsCount'].mean()],
            'Correlação (r)': [corr_com_site, corr_sem_site],
            'N (amostra)': [len(df_com_site), len(df_sem_site)]
        })

        # Stop Condition 6: Formatar o resumo
        resumo['Média totalScore'] = resumo['Média totalScore'].round(2)
        resumo['Média reviewsCount'] = resumo['Média reviewsCount'].round(2)
        resumo['Correlação (r)'] = resumo['Correlação (r)'].round(4)

        resumo.to_excel(writer, sheet_name='Resumo', index=False)

    print(f"\n✓ Arquivo '{EXCEL_OUTPUT_FILE}' gerado com sucesso!")

except ImportError:
    print(f"\nERRO: A biblioteca 'openpyxl' é necessária para salvar o arquivo Excel.")
    print("Por favor, instale-a usando: pip install openpyxl")
except PermissionError:
    print(f"\nERRO: Permissão negada para salvar o arquivo '{EXCEL_OUTPUT_FILE}'.")
    print("Verifique se o arquivo não está aberto em outro programa ou se você tem permissão de escrita.")
except Exception as e:
    print(f"\nERRO ao gerar o arquivo Excel '{EXCEL_OUTPUT_FILE}': {e}")
