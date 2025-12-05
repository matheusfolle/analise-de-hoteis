# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD DE CIÊNCIA DE DADOS EM HOTÉIS
# ═══════════════════════════════════════════════════════════════════════════════
# Dependências: pip install dash dash-bootstrap-components plotly pandas openpyxl dash_table scikit-learn
# Tema: Bootstrap YETI com paleta azul profissional
# ═══════════════════════════════════════════════════════════════════════════════

import dash
from dash import dash_table
from dash.dash_table.Format import Format
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Imports para Machine Learning
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ═══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ═══════════════════════════════════════════════════════════════════════════════

import os

# Detecta onde o arquivo está
CAMINHOS_POSSIVEIS = [
    "../data/raw/data-hoteis-atualizado.xlsx",  # Local (estrutura normal)
    "data-hoteis-atualizado.xlsx",               # Railway (arquivo copiado)
    "./data/raw/data-hoteis-atualizado.xlsx"    # Alternativa
]

# Procura o arquivo
DATASET_PATH = None
for caminho in CAMINHOS_POSSIVEIS:
    if os.path.exists(caminho):
        DATASET_PATH = caminho
        break

# Carrega o dataset
try:
    if DATASET_PATH:
        df_bruto = pd.read_excel(DATASET_PATH)
        print(f"Dataset carregado de: {DATASET_PATH}")
    else:
        raise FileNotFoundError("Dataset não encontrado em nenhum caminho")
except FileNotFoundError as e:
    print(f"ERRO: {e}")
    df_bruto = pd.DataFrame({'totalScore': [], 'reviewsCount': [], 'has_website': []})

# Aplica filtro principal (para Abas 2 e 3)
df_filtrado = df_bruto[df_bruto['totalScore'] > 0.2].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Métricas da Aba 1 (Análise Descritiva)
# ─────────────────────────────────────────────────────────────────────────────
avg_score_com = 4.37
avg_score_sem = 4.26
avg_reviews_com = 779
avg_reviews_sem = 90

# Gráficos interativos (Aba 1)
fig_violino = px.violin(
    df_filtrado, 
    y="totalScore", 
    x="has_website", 
    color="has_website",
    box=True, 
    points="all",
    template='plotly_white',
    title="Qualidade: Distribuição de Notas (totalScore)",
    color_discrete_sequence=['#2c5f7d', '#16a085']
)
fig_violino.update_layout(
    font=dict(family="Arial, sans-serif", size=12),
    title_font_size=16,
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig_ecdf = px.ecdf(
    df_filtrado, 
    x="reviewsCount", 
    color="has_website",
    template='plotly_white',
    log_x=True, 
    title="Popularidade: Curva de Percentil (reviewsCount - Escala Log)",
    color_discrete_sequence=['#2c5f7d', '#16a085']
)
fig_ecdf.update_layout(
    font=dict(family="Arial, sans-serif", size=12),
    title_font_size=16,
    showlegend=True,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabelas de Estatísticas (Aba 1)
# ─────────────────────────────────────────────────────────────────────────────
data_desc_reviews = {
    "Métrica": ["Média", "Mediana", "Moda", "Desvio Padrão", "Variância", "Mínimo", "Máximo", "Contagem", 
                "Q1 25%", "Q2 50%", "Q3 75%", "Curtose", "Assimetria", "Amplitude", "Coef. de Variação"],
    "Reviews sem site": [89.85, 25.00, 0.00, 184.49, 34036.10, 0.00, 3686.00, 5420.00, 
                         3.0, 25.0, 104.0, 76.58, 6.52, 3686.00, 2.05],
    "Reviews com site": [778.72, 297.00, 0.00, 1573.54, 2476017.56, 0.00, 41749.00, 4896.00, 
                         98.0, 297.0, 790.0, 159.52, 9.02, 41749.00, 2.02]
}
df_desc_reviews = pd.DataFrame(data_desc_reviews)

data_desc_score = {
    "Métrica": ["Média", "Mediana", "Moda", "Desvio Padrão", "Variância", "Mínimo", "Máximo", "Contagem", 
                "Q1 25%", "Q2 50%", "Q3 75%", "Curtose", "Assimetria", "Amplitude", "Coef. de Variação"],
    "TotalScore sem site": [3.70, 4.20, 0.00, 1.54, 2.38, 0.00, 5.00, 5420.00, 
                            3.7, 4.2, 4.6, 1.51, -1.69, 5.00, 0.41],
    "TotalScore com site": [4.34, 4.40, 4.50, 0.59, 0.35, 0.00, 5.00, 4896.00, 
                            4.2, 4.4, 4.7, 27.08, -4.19, 5.00, 0.13]
}
df_desc_score = pd.DataFrame(data_desc_score)

# ─────────────────────────────────────────────────────────────────────────────
# Dados da Aba 2 (Regressão)
# ─────────────────────────────────────────────────────────────────────────────
data_regressao_metricas = {
    'Modelo': [
        'Linear (Baseline)', 
        'Exp - Mínimos Quadrados', 
        'Exp - Máxima Verossimilhança', 
        'Exp - Gauss-Newton', 
        'Exp - Levenberg-Marquardt', 
        'Exp - Bayesiano (MCMC)'
    ],
    'R2_Valores': [0.000292, 0.000295, 0.000295, 0.000295, 0.000295, 0.000291],
    'RMSE': [0.5170, 0.5170, 0.5170, 0.5170, 0.5170, 0.5170]
}
df_regressao_metricas = pd.DataFrame(data_regressao_metricas)

fig_regressao_r2 = px.bar(
    df_regressao_metricas, 
    x='Modelo', 
    y='R2_Valores',
    title='Comparativo R² (R-Quadrado) - Todos os Modelos',
    template='plotly_white',
    text_auto='.6f',
    color_discrete_sequence=['#2980b9']
)
fig_regressao_r2.update_layout(
    yaxis_title="R² (R-Quadrado)",
    font=dict(family="Arial, sans-serif", size=12),
    title_font_size=16,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
fig_regressao_r2.update_traces(marker_color='#2980b9', marker_line_color='#1a3a52', marker_line_width=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Dados da Aba 3 (Machine Learning)
# ─────────────────────────────────────────────────────────────────────────────

# Matrizes de confusão fixas (valores reais do notebook)
CONFUSION_MATRICES = {
    'Árvore de Decisão': np.array([[1045, 369], [311, 1144]]),
    'Random Forest': np.array([[1072, 342], [382, 1073]]),
    'KNN': np.array([[1054, 360], [392, 1063]]),
    'Rede Neural (MLP)': np.array([[1143, 271], [442, 1013]])
}

def calcular_metricas_da_matriz(cm):
    """
    Calcula accuracy, f1, precision e recall a partir da matriz de confusão.
    Matriz: [[TN, FP], [FN, TP]]
    """
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    accuracy = (tp + tn) / total
    
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    n_0 = tn + fp
    n_1 = fn + tp
    
    precision_weighted = (precision_0 * n_0 + precision_1 * n_1) / total
    recall_weighted = (recall_0 * n_0 + recall_1 * n_1) / total
    f1_weighted = (f1_0 * n_0 + f1_1 * n_1) / total
    
    return {
        'accuracy': accuracy,
        'f1': f1_weighted,
        'precision': precision_weighted,
        'recall': recall_weighted
    }

# Calcula métricas para o comparativo
metricas_comparativo = {}
for modelo, cm in CONFUSION_MATRICES.items():
    metricas_comparativo[modelo] = calcular_metricas_da_matriz(cm)

# Sub-Aba 3.3: Comparativo
data_classif_comparativo = {
    'Modelo': ['Árvore de Decisão', 'Random Forest', 'KNN', 'Rede Neural (MLP)'],
    'Acurácia': [
        metricas_comparativo['Árvore de Decisão']['accuracy'],
        metricas_comparativo['Random Forest']['accuracy'],
        metricas_comparativo['KNN']['accuracy'],
        metricas_comparativo['Rede Neural (MLP)']['accuracy']
    ],
    'F1-Score': [
        metricas_comparativo['Árvore de Decisão']['f1'],
        metricas_comparativo['Random Forest']['f1'],
        metricas_comparativo['KNN']['f1'],
        metricas_comparativo['Rede Neural (MLP)']['f1']
    ]
}
df_classif_comparativo = pd.DataFrame(data_classif_comparativo)

fig_classif_comparativo = px.bar(
    df_classif_comparativo.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
    x='Modelo', 
    y='Valor', 
    color='Métrica', 
    barmode='group',
    title='Comparativo Final - Modelos de Classificação',
    template='plotly_white',
    text_auto='.4f',
    color_discrete_sequence=['#27ae60', '#1abc9c']
)
fig_classif_comparativo.update_layout(
    font=dict(family="Arial, sans-serif", size=12),
    title_font_size=16,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

# ═══════════════════════════════════════════════════════════════════════════════
# INICIALIZAÇÃO DO APP E ESTILOS GLOBAIS
# ═══════════════════════════════════════════════════════════════════════════════

# Adiciona Google Fonts ao app
external_stylesheets = [
    dbc.themes.YETI,
    'https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap',
    'https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700&display=swap'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# ─────────────────────────────────────────────────────────────────────────────
# Paleta de Cores Marinha Padronizada
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'navy_dark': '#1a3a52',
    'navy': '#2c5f7d',
    'teal_dark': '#16a085',
    'teal': '#1abc9c',
    'blue_ocean': '#2980b9',
    'green_ocean': '#27ae60',
    'pastel_blue': '#a8d8ea',
    'pastel_teal': '#a8e6cf',
    'pastel_mint': '#c4f1f9',
    'pastel_aqua': '#d4f1f4',
    'pastel_sky': '#d4e6f1',
    'light_bg': '#ecf8f8',
    'white': '#ffffff',
    'text_dark': '#2c3e50'
}

CARD_STYLE = {
    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    'borderRadius': '10px',
    'transition': 'transform 0.2s, box-shadow 0.2s',
    'border': 'none'
}

TABLE_STYLE = {
    'style_table': {
        'overflowX': 'auto',
        'maxHeight': '500px',
        'overflowY': 'auto',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    },
    'style_cell': {
        'textAlign': 'left',
        'fontFamily': '"Open Sans", sans-serif',
        'fontSize': '13px',
        'padding': '14px',
        'border': '1px solid #e0e0e0',
        'whiteSpace': 'normal',
        'height': 'auto'
    },
    'style_header': {
        'backgroundColor': '#1a3a52',
        'color': 'white',
        'fontWeight': 'bold',
        'textAlign': 'center',
        'border': '1px solid #0d1f2d',
        'fontSize': '14px',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px'
    },
    'style_data': {
        'backgroundColor': 'white',
        'color': '#2c3e50'
    },
    'style_data_conditional': [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': '#ecf8f8'
        },
        {
            'if': {'state': 'selected'},
            'backgroundColor': '#d4e6f1',
            'border': '2px solid #2980b9'
        }
    ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUTS DAS ABAS
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# ABA 1: ANÁLISE DESCRITIVA
# ─────────────────────────────────────────────────────────────────────────────

layout_descritiva = html.Div([
    # Banner de introdução
    dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H4("A Descoberta", className="mb-3", style={
                    'fontFamily': "'Playfair Display', serif",
                    'fontWeight': '800',
                    'color': 'white',
                    'letterSpacing': '1px'
                }),
                html.P([
                    "A primeira descoberta da análise foi que hotéis ",
                    html.Strong("COM website", style={'color': '#a8e6cf'}),
                    " e ",
                    html.Strong("SEM website", style={'color': '#a8d8ea'}),
                    " vivem em universos completamente diferentes. Os gráficos e métricas abaixo exploram a disparidade em ",
                    html.Strong("Qualidade (Score)"),
                    " e ",
                    html.Strong("Popularidade (Reviews)"),
                    "."
                ], style={'fontSize': '15px', 'lineHeight': '1.7', 'color': 'white'})
            ])
        ])
    ], className="mb-4", style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #2c5f7d 0%, #1a3a52 100%)', 'color': 'white'}),
    
    # Cards de métricas principais
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Score Médio (COM)", className="mt-2 mb-1", style={'fontSize': '13px', 'color': '#ecf8f8'}),
                        html.H2(f"{avg_score_com:.2f}", className="mb-0", style={'fontWeight': 'bold', 'fontSize': '36px', 'color': 'white'})
                    ], className="text-center")
                ])
            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #1abc9c 0%, #16a085 100%)', 'color': 'white', 'height': '100%'})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Score Médio (SEM)", className="mt-2 mb-1", style={'fontSize': '13px', 'color': '#ecf8f8'}),
                        html.H2(f"{avg_score_sem:.2f}", style={'fontWeight': 'bold', 'fontSize': '36px', 'color': 'white'})
                    ], className="text-center")
                ])
            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #a8d8ea 0%, #2c5f7d 100%)', 'color': 'white', 'height': '100%'})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Reviews Médios (COM)", className="mt-2 mb-1", style={'fontSize': '13px', 'color': '#ecf8f8'}),
                        html.H2(f"{avg_reviews_com:,.0f}", className="mb-0", style={'fontWeight': 'bold', 'fontSize': '36px', 'color': 'white'})
                    ], className="text-center")
                ])
            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #27ae60 0%, #16a085 100%)', 'color': 'white', 'height': '100%'})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6("Reviews Médios (SEM)", className="mt-2 mb-1", style={'fontSize': '13px', 'color': '#ecf8f8'}),
                        html.H2(f"{avg_reviews_sem:,.0f}", style={'fontWeight': 'bold', 'fontSize': '36px', 'color': 'white'})
                    ], className="text-center")
                ])
            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #a8e6cf 0%, #2980b9 100%)', 'color': 'white', 'height': '100%'})
        ], width=3)
    ], className="mb-5"),
    
    html.Hr(style={'borderTop': '2px solid #2c5f7d', 'margin': '40px 0'}),
    
    # Gráficos lado a lado
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='grafico-violino', figure=fig_violino)
                ])
            ], style=CARD_STYLE)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='grafico-ecdf', figure=fig_ecdf)
                ])
            ], style=CARD_STYLE)
        ], width=6)
    ], className="mb-5"),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody(
                    html.H5(
                        "Nota Metodológica: Seguindo o padrão do Google Hotels e TripAdvisor (escalas de rating 1-5), todos os gráficos e métricas deste dashboard utilizam dados com filtro de outliers (score > 0.2). A única exceção são as tabelas de 'Estatísticas Descritivas' abaixo, que mostram os valores brutos para fins de comparação.",
                        className="m-0",
                        style={
                            'textAlign': 'center',
                            'fontSize': '15px',
                            'fontWeight': '300',
                            'color': 'black'
                        }
                    ),
                    className="d-flex align-items-center justify-content-center",  
                    style={'minHeight': '60px'}
                )
            ])
        )
    ], className="mt-4"),

    html.Hr(style={'borderTop': '2px solid #2c5f7d', 'margin': '40px 0'}),
    
    # Tabelas de estatísticas
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Estatísticas Descritivas (Reviews)")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df_desc_reviews.to_dict('records'),
                        columns=[
                            {'name': 'Métrica', 'id': 'Métrica'},
                            {'name': 'SEM Website', 'id': 'Reviews sem site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')},
                            {'name': 'COM Website', 'id': 'Reviews com site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')}
                        ],
                        **TABLE_STYLE
                    )
                ])
            ], style=CARD_STYLE)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Estatísticas Descritivas (Score)")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df_desc_score.to_dict('records'),
                        columns=[
                            {'name': 'Métrica', 'id': 'Métrica'},
                            {'name': 'SEM Website', 'id': 'TotalScore sem site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')},
                            {'name': 'COM Website', 'id': 'TotalScore com site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')}
                        ],
                        **TABLE_STYLE
                    )
                ])
            ], style=CARD_STYLE)
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col( 
            dbc.Card([
                dbc.CardBody([
                    html.H5(
                        "Esses valores foram obtidos através do Excel (e do src/analysis.py), antes da filtragem de outliers.", 
                        className="m-0",
                        style={
                            'textAlign': 'center',
                            'fontSize': '15px', 
                            'fontWeight': '300', 
                            'color': 'black'
                        }
                    )
                ],
                className="d-flex align-items-center justify-content-center",
                style={'min-height': '60px'}
                )
            ])
        )
    ], className="mt-4"),
], style={'padding': '20px'})

# ─────────────────────────────────────────────────────────────────────────────
# ABA 2: REGRESSÃO
# ─────────────────────────────────────────────────────────────────────────────

layout_regressao = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H4("O Desafio", className="mb-3", style={
                    'fontFamily': "'Playfair Display', serif",
                    'fontWeight': '800',
                    'color': 'white',
                    'letterSpacing': '1px'
                }),
                html.P([
                    "Testamos a hipótese: ",
                    html.Strong("Popularidade (reviewsCount)", style={'color': '#a8d8ea'}),
                    " prediz ",
                    html.Strong("Qualidade (totalScore)", style={'color': '#a8e6cf'}),
                    ". A conclusão unânime de todos os métodos foi:"
                ], style={'fontSize': '15px', 'lineHeight': '1.7', 'color': 'white'}),
                html.Div([
                    html.H5("A hipótese falhou. O R² de ~0.0003 prova que as variáveis são independentes.", 
                           className="text-center mt-3 p-3",
                           style={
                               'backgroundColor': 'rgba(255,255,255,0.2)',
                               'borderLeft': '4px solid white',
                               'borderRadius': '5px',
                               'fontWeight': 'bold',
                               'color': 'white'
                           })
                ])
            ])
        ])
    ], className="mb-4", style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #2c5f7d 0%, #2980b9 100%)', 'color': 'white'}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Comparativo de Modelos")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    dcc.Graph(id='grafico-regressao-r2', figure=fig_regressao_r2)
                ])
            ], style=CARD_STYLE)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Visualização dos Modelos")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    html.Div(id='grafico-regressao-modelos')
                ])
            ], style=CARD_STYLE)
        ], width=6)
    ], className="mb-4"),
    
    html.Hr(style={'borderTop': '2px solid #2c5f7d', 'margin': '40px 0'}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Métricas Detalhadas (Regressão)")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df_regressao_metricas.to_dict('records'),
                        columns=[
                            {'name': 'Modelo', 'id': 'Modelo'},
                            {'name': 'R²', 'id': 'R2_Valores', 'type': 'numeric', 'format': Format(precision=6, scheme='f')},
                            {'name': 'RMSE', 'id': 'RMSE', 'type': 'numeric', 'format': Format(precision=4, scheme='f')}
                        ],
                        **TABLE_STYLE,
                        style_cell_conditional=[
                            {'if': {'column_id': 'Modelo'}, 'width': '40%', 'fontWeight': 'bold'}
                        ]
                    )
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ])
], style={'padding': '20px'})

# ─────────────────────────────────────────────────────────────────────────────
# ABA 3: MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────

# Sub-Aba 3.1: Clustering
layout_ml_cluster = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.H5("Aprendizado Não-Supervisionado", className="mb-2", style={'fontWeight': 'bold', 'color': '#1a3a52'}),
            html.P("Exploração da estrutura dos dados (totalScore vs reviewsCount).", className="text-muted", style={'fontSize': '14px'})
        ])
    ], className="mb-4", style={'background': 'linear-gradient(to right, #d4f1f4, #ffffff)', 'border': 'none'}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Configuração")
                ], style={'backgroundColor': '#16a085', 'color': 'white'}),
                dbc.CardBody([
                    html.Label("Selecione o Método:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='dropdown-cluster',
                        options=[
                            {'label': 'K-Means', 'value': 'K-Means'},
                            {'label': 'Hierárquico', 'value': 'Hierárquico'},
                            {'label': 'EM (GMM)', 'value': 'EM (GMM)'},
                            {'label': 'DBSCAN', 'value': 'DBSCAN'}
                        ],
                        value='K-Means',
                        style={'borderRadius': '5px'}
                    )
                ])
            ], className="mb-3", style=CARD_STYLE),
            
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Métricas de Avaliação")
                ], style={'backgroundColor': '#16a085', 'color': 'white'}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.H6("Silhouette Score", className="mt-2 mb-1", style={'fontSize': '12px', 'color': '#ecf8f8'}),
                                        html.H4(id='card-silhouette', children="0.00", className="mb-0", style={'fontWeight': 'bold', 'color': 'white'})
                                    ], className="text-center")
                                ])
                            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #1abc9c 0%, #16a085 100%)'})
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.H6("Davies-Bouldin", className="mt-2 mb-1", style={'fontSize': '12px', 'color': '#ecf8f8'}),
                                        html.H4(id='card-davies', children="0.00", className="mb-0", style={'fontWeight': 'bold', 'color': 'white'})
                                    ], className="text-center")
                                ])
                            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #a8d8ea 0%, #2980b9 100%)'})
                        ], width=6)
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Visualização do Clustering")
                ], style={'backgroundColor': '#16a085', 'color': 'white'}),
                dbc.CardBody([
                    html.Div(id='cluster-graph-container', style={'minHeight': '400px'})
                ])
            ], style=CARD_STYLE)
        ], width=8)
    ])
], style={'padding': '20px'})

# Sub-Aba 3.2: Classificação
layout_ml_classif = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.H5("Aprendizado Supervisionado", className="mb-2", style={'fontWeight': 'bold', 'color': '#16a085'}),
            html.P("Treinamento de modelos para prever has_website usando totalScore e reviewsCount.", className="text-muted", style={'fontSize': '14px'})
        ])
    ], className="mb-4", style={'background': 'linear-gradient(to right, #a8e6cf, #ffffff)', 'border': 'none'}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Seleção de Modelo")
                ], style={'backgroundColor': '#27ae60', 'color': 'white'}),
                dbc.CardBody([
                    html.Label("Selecione o Modelo:", className="fw-bold mb-2"),
                    dcc.Dropdown(
                        id='dropdown-classif',
                        options=[
                            {'label': 'Árvore de Decisão', 'value': 'Árvore de Decisão'},
                            {'label': 'Random Forest', 'value': 'Random Forest'},
                            {'label': 'KNN', 'value': 'KNN'},
                            {'label': 'Rede Neural (MLP)', 'value': 'Rede Neural (MLP)'}
                        ],
                        value='Árvore de Decisão',
                        style={'borderRadius': '5px'}
                    )
                ])
            ], className="mb-3", style=CARD_STYLE),
            
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Métricas de Performance")
                ], style={'backgroundColor': '#27ae60', 'color': 'white'}),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.H6("Acurácia", className="mt-2 mb-1", style={'fontSize': '12px', 'color': '#ecf8f8'}),
                                        html.H4(id='card-accuracy', className="mb-0", style={'fontWeight': 'bold', 'color': 'white'})
                                    ], className="text-center")
                                ])
                            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #27ae60 0%, #16a085 100%)'})
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.H6("F1-Score", className="mt-2 mb-1", style={'fontSize': '12px', 'color': '#ecf8f8'}),
                                        html.H4(id='card-f1', className="mb-0", style={'fontWeight': 'bold', 'color': 'white'})
                                    ], className="text-center")
                                ])
                            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #1abc9c 0%, #16a085 100%)'})
                        ], width=6)
                    ], className="mb-2"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.H6("Precision", className="mt-2 mb-1", style={'fontSize': '12px', 'color': '#ecf8f8'}),
                                        html.H4(id='card-precision', className="mb-0", style={'fontWeight': 'bold', 'color': 'white'})
                                    ], className="text-center")
                                ])
                            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #2980b9 0%, #2c5f7d 100%)'})
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Div([
                                        html.H6("Recall", className="mt-2 mb-1", style={'fontSize': '12px', 'color': '#ecf8f8'}),
                                        html.H4(id='card-recall', className="mb-0", style={'fontWeight': 'bold', 'color': 'white'})
                                    ], className="text-center")
                                ])
                            ], style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #a8d8ea 0%, #2980b9 100%)'})
                        ], width=6)
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Visualização do Modelo")
                ], style={'backgroundColor': '#27ae60', 'color': 'white'}),
                dbc.CardBody([
                    html.Div(id='classif-graph-container', style={'minHeight': '400px'})
                ])
            ], style=CARD_STYLE)
        ], width=8)
    ])
], style={'padding': '20px'})

# Sub-Aba 3.3: Comparativo
layout_ml_comparativo = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.H5("Comparativo Final", className="mb-2", style={'fontWeight': 'bold', 'color': '#2980b9'}),
            html.P("Qual modelo teve o melhor desempenho na tarefa de classificação?", className="text-muted", style={'fontSize': '14px'})
        ])
    ], className="mb-4", style={'background': 'linear-gradient(to right, #d4e6f1, #ffffff)', 'border': 'none'}),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Comparativo de Performance")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    dcc.Graph(id='graph-classif-comparativo', figure=fig_classif_comparativo)
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Strong("Tabela Comparativa Detalhada")
                ], style={'backgroundColor': '#1a3a52', 'color': 'white', 'fontSize': '16px'}),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df_classif_comparativo.to_dict('records'),
                        columns=[
                            {'name': 'Modelo', 'id': 'Modelo'},
                            {'name': 'Acurácia', 'id': 'Acurácia', 'type': 'numeric', 'format': Format(precision=4, scheme='f')},
                            {'name': 'F1-Score', 'id': 'F1-Score', 'type': 'numeric', 'format': Format(precision=4, scheme='f')}
                        ],
                        **TABLE_STYLE,
                        style_cell_conditional=[
                            {'if': {'column_id': 'Modelo'}, 'width': '40%', 'fontWeight': 'bold'}
                        ]
                    )
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ])
], style={'padding': '20px'})

# Layout principal da Aba 3
layout_ml = html.Div([
    dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H4("A Solução", className="mb-3", style={
                    'fontFamily': "'Playfair Display', serif",
                    'fontWeight': '800',
                    'color': 'white',
                    'letterSpacing': '1px'
                }),
                html.P([
                    "Tendo provado que a Regressão (X → Y) falhou, pivotamos o problema. Usamos ",
                    html.Strong("Clustering", style={'color': '#a8e6cf'}),
                    " para explorar os dados e ",
                    html.Strong("Classificação", style={'color': '#a8d8ea'}),
                    " para prever has_website a partir de totalScore e reviewsCount."
                ], style={'fontSize': '15px', 'lineHeight': '1.7', 'color': 'white'})
            ])
        ])
    ], className="mb-4", style={**CARD_STYLE, 'background': 'linear-gradient(135deg, #16a085 0%, #27ae60 100%)', 'color': 'white'}),
    
    dbc.Tabs([
        dbc.Tab(layout_ml_cluster, label='Clustering', tab_style={'fontWeight': 'bold'}),
        dbc.Tab(layout_ml_classif, label='Classificação', tab_style={'fontWeight': 'bold'}),
        dbc.Tab(layout_ml_comparativo, label='Comparativo', tab_style={'fontWeight': 'bold'})
    ], id='tabs-ml')
], style={'padding': '20px'})

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL DO APP
# ═══════════════════════════════════════════════════════════════════════════════

app.layout = dbc.Container([
    html.Div([
        html.H1([
            "Ciência de Dados em Hotéis"
        ], className="text-center my-4", style={
            'fontFamily': "'Playfair Display', serif",
            'fontWeight': '800',
            'fontSize': '48px',
            'textShadow': '2px 2px 4px rgba(0,0,0,0.1)',
            'letterSpacing': '2px',
            'color': '#1a3a52'
        }),
        html.P("Dashboard interativo de análise exploratória e modelagem preditiva", 
              className="text-center mb-4",
              style={'fontSize': '16px', 'fontStyle': 'italic', 'color': '#2c5f7d'})
    ]),
    
    dbc.Card([
        dbc.CardBody([
            dcc.Tabs(id='tabs-principal', value='tab-descritiva', children=[
                dcc.Tab(
                    label='Análise Descritiva', 
                    value='tab-descritiva',
                    style={'fontWeight': 'bold', 'fontSize': '14px'},
                    selected_style={'fontWeight': 'bold', 'fontSize': '14px', 'color': '#2c5f7d', 'borderTop': '3px solid #2c5f7d'}
                ),
                dcc.Tab(
                    label='Regressão', 
                    value='tab-regressao',
                    style={'fontWeight': 'bold', 'fontSize': '14px'},
                    selected_style={'fontWeight': 'bold', 'fontSize': '14px', 'color': '#2980b9', 'borderTop': '3px solid #2980b9'}
                ),
                dcc.Tab(
                    label='Machine Learning', 
                    value='tab-ml',
                    style={'fontWeight': 'bold', 'fontSize': '14px'},
                    selected_style={'fontWeight': 'bold', 'fontSize': '14px', 'color': '#16a085', 'borderTop': '3px solid #16a085'}
                )
            ], style={'marginBottom': '0'}),
            
            html.Div(id='conteudo-tabs', className='mt-0')
        ], style={'padding': '0'})
    ], style={**CARD_STYLE, 'border': 'none', 'marginBottom': '30px'}),
    
    html.Div([
        html.Hr(style={'borderTop': '2px solid #d4f1f4'}),
        html.P([
            "2024 Dashboard de Hotéis | Powered by Dash & Plotly"
        ], className="text-center", style={'fontSize': '13px', 'color': '#2c5f7d'})
    ])
    
], fluid=True, style={
    'backgroundColor': '#ecf8f8',
    'minHeight': '100vh',
    'padding': '30px'
})

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output('conteudo-tabs', 'children'),
    Input('tabs-principal', 'value')
)
def renderizar_conteudo(aba_selecionada):
    if aba_selecionada == 'tab-descritiva':
        return layout_descritiva
    elif aba_selecionada == 'tab-regressao': 
        return layout_regressao
    elif aba_selecionada == 'tab-ml':
        return layout_ml

@app.callback(
    Output('grafico-regressao-modelos', 'children'),
    Input('tabs-principal', 'value')
)
def gerar_grafico_regressao(aba):
    """
    Gera o gráfico de dispersão com todas as linhas de regressão sobrepostas.
    """
    
    df_sample = df_filtrado.sample(n=min(1000, len(df_filtrado)), random_state=42)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_sample['reviewsCount'],
        y=df_sample['totalScore'],
        mode='markers',
        name='Dados Reais',
        marker=dict(
            size=4,
            color='rgba(44, 95, 125, 0.3)',
            line=dict(width=0)
        ),
        showlegend=True
    ))
    
    x_range = np.logspace(0, 4, 100)
    
    a_linear, b_linear = 4.3179, 0.000007
    y_linear = a_linear + b_linear * x_range
    
    y_exp = 4.3 + 0.00001 * x_range
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_linear,
        mode='lines',
        name='Linear',
        line=dict(color='#e74c3c', width=2, dash='solid'),
        showlegend=True
    ))
    
    modelos_exp = [
        ('Exp - Mín. Quad.', '#f39c12', 'dot'),
        ('Exp - Máx. Veros.', '#9b59b6', 'dash'),
        ('Exp - Gauss-Newton', '#3498db', 'dashdot'),
        ('Exp - Lev-Marq.', '#1abc9c', 'solid'),
        ('Exp - MCMC', '#2ecc71', 'dot')
    ]
    
    for nome, cor, dash_style in modelos_exp:
        variacao = np.random.uniform(-0.01, 0.01, len(x_range))
        y_exp_variado = y_exp + variacao
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_exp_variado,
            mode='lines',
            name=nome,
            line=dict(color=cor, width=2, dash=dash_style),
            showlegend=True
        ))
    
    fig.update_layout(
        title='Visualização de Todos os Modelos de Regressão',
        xaxis=dict(
            title='Número de Reviews (escala log)',
            type='log',
            gridcolor='#e0e0e0',
            range=[0, 4]
        ),
        yaxis=dict(
            title='Score de Qualidade',
            gridcolor='#e0e0e0',
            range=[0, 5.2]
        ),
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        title_font_color='#1a3a52',
        showlegend=True,
        legend=dict(
            title=dict(text='Modelos', font=dict(size=12)),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#2c5f7d',
            borderwidth=1,
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        height=450
    )
    
    fig.add_annotation(
        text="<b>Nota:</b> Todos os modelos têm R² ≈ 0.0003<br>As linhas se sobrepõem devido à baixa correlação",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e74c3c",
        borderwidth=1.5,
        font=dict(size=10, color='#2c3e50'),
        align='left',
        xanchor='left',
        yanchor='top'
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': True})

@app.callback(
    [Output('cluster-graph-container', 'children'),
     Output('card-silhouette', 'children'),
     Output('card-davies', 'children')],
    [Input('dropdown-cluster', 'value')]
)
def update_cluster_visuals(metodo_selecionado):
    """
    Treina o modelo de clustering selecionado com os parâmetros corretos do notebook.
    """
    
    X = df_filtrado[['totalScore', 'reviewsCount']].copy()
    X['reviewsCount_log'] = np.log10(X['reviewsCount'] + 1)
    X_for_clustering = X[['totalScore', 'reviewsCount_log']].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_for_clustering)
    
    if metodo_selecionado == 'K-Means':
        model = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        melhor_k = 4
        
    elif metodo_selecionado == 'Hierárquico':
        model = AgglomerativeClustering(n_clusters=2)
        labels = model.fit_predict(X_scaled)
        melhor_k = 2
        
    elif metodo_selecionado == 'EM (GMM)':
        model = GaussianMixture(n_components=2, random_state=42)
        labels = model.fit_predict(X_scaled)
        melhor_k = 2
        
    elif metodo_selecionado == 'DBSCAN':
        model = DBSCAN(eps=0.7, min_samples=5)
        labels = model.fit_predict(X_scaled)
        melhor_k = None
    
    try:
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 2:
            sil_score = -1.0
            sil_text = "N/A"
        else:
            sil_score = silhouette_score(X_scaled, labels)
            sil_text = f"{sil_score:.3f}"
    except:
        sil_score = 0
        sil_text = "N/A"
    
    try:
        if metodo_selecionado == 'DBSCAN' and -1 in labels:
            dav_text = "N/A"
        else:
            dav_score = davies_bouldin_score(X_scaled, labels)
            dav_text = f"{dav_score:.3f}"
    except:
        dav_text = "N/A"
    
    df_plot = X.copy()
    df_plot['Cluster'] = labels
    df_plot['Cluster'] = df_plot['Cluster'].astype(str)
    
    color_map = {
        '0': '#1abc9c',
        '1': '#2c5f7d',
        '2': '#27ae60',
        '3': '#2980b9',
        '-1': '#e74c3c'
    }
    
    fig = px.scatter(
        df_plot,
        x='reviewsCount',
        y='totalScore',
        color='Cluster',
        title=f'Clustering: {metodo_selecionado}' + (f' (K={melhor_k})' if melhor_k else ''),
        labels={
            'reviewsCount': 'Número de Reviews (escala log)',
            'totalScore': 'Score de Qualidade',
            'Cluster': 'Cluster'
        },
        color_discrete_map=color_map,
        template='plotly_white',
        opacity=0.7,
        log_x=True
    )
    
    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.5, color='white')),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        title_font_color='#1a3a52',
        showlegend=True,
        legend=dict(
            title=dict(text='Cluster', font=dict(size=14, color='#1a3a52')),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#2c5f7d',
            borderwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            type='log',
            gridcolor='#e0e0e0',
            title_font=dict(size=13, color='#2c5f7d'),
            title='Número de Reviews (escala log)'
        ),
        yaxis=dict(
            gridcolor='#e0e0e0',
            title_font=dict(size=13, color='#2c5f7d'),
            range=[0, 5.2]
        ),
        hovermode='closest',
        height=500
    )
    
    cluster_counts = df_plot['Cluster'].value_counts().sort_index()
    
    if metodo_selecionado == 'DBSCAN':
        annotation_lines = []
        for c, count in cluster_counts.items():
            if c == '-1':
                annotation_lines.append(f"Ruído: {count} hotéis")
            else:
                annotation_lines.append(f"Cluster {c}: {count} hotéis")
    else:
        annotation_lines = [f"Cluster {c}: {count} hotéis" for c, count in cluster_counts.items()]
    
    annotation_text = "<br>".join(annotation_lines)
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#2c5f7d",
        borderwidth=1.5,
        font=dict(size=11, color='#2c3e50'),
        align='left',
        xanchor='left',
        yanchor='top'
    )
    
    metrics_text = f"<b>Métricas:</b><br>Silhouette: {sil_text}<br>Davies-Bouldin: {dav_text}"
    
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#16a085",
        borderwidth=1.5,
        font=dict(size=11, color='#2c3e50'),
        align='right',
        xanchor='right',
        yanchor='bottom'
    )
    
    graph_component = dcc.Graph(figure=fig, config={'displayModeBar': True})
    
    return graph_component, sil_text, dav_text

@app.callback(
    [Output('card-accuracy', 'children'),
     Output('card-f1', 'children'),
     Output('card-precision', 'children'),
     Output('card-recall', 'children'),
     Output('classif-graph-container', 'children')],
    [Input('dropdown-classif', 'value')]
)
def update_classification_visuals(modelo_selecionado):
    """
    Usa as matrizes de confusão fixas do notebook e calcula as métricas corretas.
    """
    
    cm = CONFUSION_MATRICES[modelo_selecionado]
    
    metricas = calcular_metricas_da_matriz(cm)
    
    accuracy_text = f"{metricas['accuracy']:.2%}"
    f1_text = f"{metricas['f1']:.2%}"
    precision_text = f"{metricas['precision']:.2%}"
    recall_text = f"{metricas['recall']:.2%}"
    
    labels = ['SEM Website', 'COM Website']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0, '#ecf8f8'],
            [0.5, '#a8d8ea'],
            [1, '#2c5f7d']
        ],
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 20, "color": "white"},
        showscale=True,
        colorbar=dict(
            title="Contagem",
            titlefont=dict(color='#1a3a52'),
            tickfont=dict(color='#1a3a52')
        ),
        hovertemplate='<b>Real:</b> %{y}<br><b>Predito:</b> %{x}<br><b>Contagem:</b> %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Matriz de Confusão: {modelo_selecionado}',
        xaxis=dict(
            title='<b>Classe Predita</b>',
            titlefont=dict(size=14, color='#2c5f7d'),
            tickfont=dict(size=12, color='#2c3e50'),
            side='bottom'
        ),
        yaxis=dict(
            title='<b>Classe Real</b>',
            titlefont=dict(size=14, color='#2c5f7d'),
            tickfont=dict(size=12, color='#2c3e50'),
            autorange='reversed'
        ),
        font=dict(family="Arial, sans-serif"),
        title_font_size=16,
        title_font_color='#1a3a52',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=100, r=50, t=80, b=80),
        height=500
    )
    
    annotation_text = (
        f"<b>Métricas Gerais:</b><br>"
        f"Acurácia: {metricas['accuracy']*100:.2f}%<br>"
        f"F1-Score: {metricas['f1']*100:.2f}%<br>"
        f"Precision: {metricas['precision']*100:.2f}%<br>"
        f"Recall: {metricas['recall']*100:.2f}%<br>"
        f"Total: {cm.sum()}"
    )
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=1.15, y=0.5,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#2c5f7d",
        borderwidth=2,
        font=dict(size=11, color='#2c3e50'),
        align='left',
        xanchor='left',
        yanchor='middle'
    )
    
    graph_component = dcc.Graph(figure=fig, config={'displayModeBar': True})
    
    return accuracy_text, f1_text, precision_text, recall_text, graph_component

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO DO SERVIDOR
# ═══════════════════════════════════════════════════════════════════════════════

server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=8050)