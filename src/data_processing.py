# 2.1 Montar Google Drive e Carregar Dataset
import pandas as pd
from google.colab import drive

print("🚀 Montando o Google Drive...")
drive.mount('/content/drive')

# 2.2 Definir o caminho e carregar o arquivo ORIGINAL
CAMINHO_ORIGINAL = '/content/drive/MyDrive/projeto-datascience/data-hoteis-atualizado.xlsx'

print(f"\nBuscando arquivo em: {CAMINHO_ORIGINAL}...")
try:
    # Carrega o dataframe BRUTO
    df_bruto = pd.read_excel(CAMINHO_ORIGINAL)
    print("✅ Arquivo BRUTO carregado com sucesso!")
    print(f"📊 Total de registros brutos: {len(df_bruto):,}")
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo não encontrado em: {CAMINHO_ORIGINAL}")
except Exception as e:
    print(f"❌ Ocorreu um erro inesperado: {e}")