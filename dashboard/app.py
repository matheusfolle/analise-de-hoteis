# --- Dependências ---
# pip install dash dash-bootstrap-components plotly pandas openpyxl dash_table
# --------------------

# Importações necessárias
import dash
import dash_table
from dash_table.Format import Format
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px  # Para os gráficos
import plotly.graph_objects as go # Para o gráfico placeholder
import pandas as pd         # Para carregar os dados

# --- Carregamento e Preparação dos Dados ---

# Carrega o dataset
try:
    df_bruto = pd.read_excel("../data/data-hoteis-atualizado.xlsx")
except FileNotFoundError:
    print("ERRO: Arquivo '../data/data-hoteis-atualizado.xlsx' não encontrado.")
    df_bruto = pd.DataFrame({'totalScore': [], 'reviewsCount': [], 'has_website': []})

# Aplica o filtro principal do projeto (para Abas 2 e 3)
df_filtrado = df_bruto[df_bruto['totalScore'] > 0.2].copy()

# Apague as linhas df_com e df_sem, já não são precisas

# Cards (Aba 1)
# Cards (Aba 1)
# --- PATCH 14 ---
# Substituímos os cálculos (.mean()) pelos valores finais do notebook.

avg_score_com = 4.34
avg_score_sem = 3.70
avg_reviews_com = 779 # (Valor do seu briefing)
avg_reviews_sem = 90  # (Valor do seu briefing)

# Gráficos (Aba 1)
fig_violino = px.violin(df_bruto, 
                        y="totalScore", 
                        x="has_website", 
                        color="has_website",
                        box=True, 
                        points="all",
                        template='seaborn', 
                        title="Qualidade: Distribuição de Notas (totalScore)")

fig_ecdf = px.ecdf(df_bruto, 
                   x="reviewsCount", 
                   color="has_website",
                   template='seaborn', 
                   log_x=True, 
                   title="Popularidade: Curva de Percentil (reviewsCount - Escala Log)")

# --- Fim da Preparação (Aba 1) ---

# ******************************************************
# --- INÍCIO - PASSO 1 (Adicionar Tabelas da Aba 1) ---

# Vamos recriar os dataframes COM/SEM, mas usando o df_bruto
# (NOTA: O seu código AINDA está a usar o 'df' filtrado globalmente,
# vamos corrigir isso para os gráficos da Aba 1 usarem o 'df_bruto'
# para bater certo com os seus cards "hard-coded")

# Tabela 1: Estatísticas de 'Reviews'
data_desc_reviews = {
    "Métrica": ["Média", "Mediana", "Moda", "Desvio Padrão", "Variância", "Mínimo", "Máximo", "Contagem", 
                "Q1 25%", "Q2 50%", "Q3 75%", "Curtose", "Assimetria", "Amplitude", "Coef. de Variação"],
    "Reviews sem site": [89.85, 25.00, 0.00, 184.49, 34036.10, 0.00, 3686.00, 5420.00, 
                         3.0, 25.0, 104.0, 76.58, 6.52, 3686.00, 2.05],
    "Reviews com site": [778.72, 297.00, 0.00, 1573.54, 2476017.56, 0.00, 41749.00, 4896.00, 
                         98.0, 297.0, 790.0, 159.52, 9.02, 41749.00, 2.02]
}
df_desc_reviews = pd.DataFrame(data_desc_reviews)

# Tabela 2: Estatísticas de 'TotalScore'
data_desc_score = {
    "Métrica": ["Média", "Mediana", "Moda", "Desvio Padrão", "Variância", "Mínimo", "Máximo", "Contagem", 
                "Q1 25%", "Q2 50%", "Q3 75%", "Curtose", "Assimetria", "Amplitude", "Coef. de Variação"],
    "TotalScore sem site": [3.70, 4.20, 0.00, 1.54, 2.38, 0.00, 5.00, 5420.00, 
                            3.7, 4.2, 4.6, 1.51, -1.69, 5.00, 0.41],
    "TotalScore com site": [4.34, 4.40, 4.50, 0.59, 0.35, 0.00, 5.00, 4896.00, 
                            4.2, 4.4, 4.7, 27.08, -4.19, 5.00, 0.13]
}
df_desc_score = pd.DataFrame(data_desc_score)

# --- FIM - PASSO 1 ---
# ******************************************************

# --- Preparação de Dados (Aba 2 - Regressão) ---

# Dados REAIS do seu notebook
data_regressao_metricas = {
    'Modelo': [
        'Linear (Baseline)', 
        'Exp - Mínimos Quadrados', 
        'Exp - Máxima Verossimilhança', 
        'Exp - Gauss-Newton', 
        'Exp - Levenberg-Marquardt', 
        'Exp - Bayesiano (MCMC)'
    ],
    'R2_Valores': [
        0.000292, 
        0.000295, 
        0.000295, 
        0.000295, 
        0.000295, 
        0.000291
    ],
    'RMSE': [
        0.516955, 
        0.516954, 
        0.516954, 
        0.516954, 
        0.516954, 
        0.516955
    ]
}
df_regressao_metricas = pd.DataFrame(data_regressao_metricas)

# Gráfico para o Comparativo R² (Gráfico 13)
fig_regressao_r2 = px.bar(df_regressao_metricas, 
                          x='Modelo', 
                          y='R2_Valores',
                          title='Comparativo R² (R-Quadrado) - Todos Modelos',
                          template='seaborn',
                          text_auto='.6f') 
fig_regressao_r2.update_layout(yaxis_title="R² (R-Quadrado)")


# --- Fim da Preparação (Aba 2) ---

# --- Fim da Preparação (Aba 2) ---

# ******************************************************
# --- INÍCIO DO NOVO CÓDIGO (PATCH 2) ---
# --- Preparação de Dados (Aba 3 - Machine Learning) ---

# Vamos criar dados fictícios (MOCK DATA) para os resultados
# Você pode substituir estes valores pelos seus resultados reais do notebook
# -----------------------------------------------------------------
# 1. DADOS DE CLUSTERING (Não-Supervisionado)
# -----------------------------------------------------------------

# Dicionário com os resultados REAIS e os NOMES DAS IMAGENS
cluster_data = {
    'K-Means': {
        'silhouette': 0.511455,
        'davies_bouldin': 0.646102, # (Ou o valor real que você encontrar)
        'img_filename': 'clustering_kmeans.png' 
    },
    'Hierárquico': {
        'silhouette': 0.678643,
        'davies_bouldin': 0.685730, # (Ou o valor real que você encontrar)
        'img_filename': 'clustering_hierarchical.png' 
    },
    'EM (GMM)': {
        'silhouette': 0.285938,
        'davies_bouldin': 1.548837,
        'img_filename': 'clustering_em.png'
    },
    'DBSCAN': {
        'silhouette': 0.780933,
        'davies_bouldin': None,
        'img_filename': 'clustering_dbscan.png'
    }
}

# -----------------------------------------------------------------
# 2. DADOS DE CLASSIFICAÇÃO (Supervisionado)
# -----------------------------------------------------------------
# Gráficos placeholder para Classificação
fig_tree_placeholder = go.Figure(layout={'title': 'Gráfico da Árvore (Substitua-me)'})
fig_roc_placeholder = px.line(x=[0, 1], y=[0, 1], title='Curva ROC (Substitua-me)', template='seaborn')
fig_cm_placeholder = px.imshow([[10, 5], [2, 12]], text_auto=True, title='Matriz de Confusão (Substitua-me)')

# Dicionário com os resultados REAIS e os NOMES DAS IMAGENS (Corrigido)
classif_data = {
    'Árvore de Decisão': {
        'accuracy': 0.7630, # <-- O seu valor REAL
        'f1': 0.7709,        # <-- O seu valor REAL
        'precision': 0.7561,   # <-- O seu valor REAL (calculado)
        'recall': 0.7862,    # <-- O seu valor REAL (calculado)
        'img_filename': 'classificacao_arvore.png' # <-- O nome do seu ficheiro
    },
    'Random Forest': {
        'accuracy': 0.7476, # <-- O seu valor REAL
        'f1': 0.7477,        # <-- O seu valor REAL
        'precision': 0.7583,   # <-- O seu valor REAL (calculado)
        'recall': 0.7375,    # <-- O seu valor REAL (calculado)
        'img_filename': 'classificacao_randomforest.png' # <-- O nome do seu ficheiro
    },
    'KNN': {
        'accuracy': 0.7379, # <-- O seu valor REAL
        'f1': 0.7387,        # <-- O seu valor REAL
        'precision': 0.7470,   # <-- O seu valor REAL (calculado)
        'recall': 0.7302,    # <-- O seu valor REAL (calculado)
        'img_filename': 'classificacao_knn.png' # <-- O nome do seu ficheiro
    },
    'Rede Neural (MLP)': {
        'accuracy': 0.7515, # <-- O seu valor REAL
        'f1': 0.7397,        # <-- O seu valor REAL
        'precision': 0.7889,   # <-- O seu valor REAL (calculado)
        'recall': 0.6962,    # <-- O seu valor REAL (calculado)
        'img_filename': 'classificacao_redeneural.png' # <-- O nome do seu ficheiro
    },
}
# -----------------------------------------------------------------
# 3. DADOS DE COMPARAÇÃO (Classificação) - AGORA COM VALORES REAIS
data_classif_comparativo = {
    'Modelo': ['Árvore de Decisão', 'Random Forest', 'KNN', 'Rede Neural (MLP)'],
    'Acurácia': [0.7630, 0.7476, 0.7379, 0.7515],
    'F1-Score': [0.7709, 0.7477, 0.7387, 0.7397]
}
df_classif_comparativo = pd.DataFrame(data_classif_comparativo)

# Gráfico para o Comparativo Final
fig_classif_comparativo = px.bar(
    df_classif_comparativo.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
    x='Modelo', 
    y='Valor', 
    color='Métrica', 
    barmode='group',
    title='Comparativo Final - Modelos de Classificação',
    template='seaborn',
    text_auto='.4f' # Mudei para 4 casas decimais para precisão
)

# Inicializa o app Dash com tema YETI
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI], suppress_callback_exceptions=True)
server = app.server # Para deploy

# --- Layout da Aba 1 (Descritiva) ---
layout_descritiva = html.Div([
    # Linha 0: O "textinho" de introdução
    dbc.Row(
        dbc.Col(
            dcc.Markdown("""
                ##### Ato 1: A Descoberta
                A primeira descoberta da análise foi que hotéis **COM website** e **SEM website** vivem em universos completamente diferentes. Os gráficos e métricas abaixo exploram
                a disparidade em **Qualidade (`Score`)** e **Popularidade (`Reviews`)**.
            """, className="mb-4")
        )
    ),
    
    # Linha 1: Métricas Principais (Cards)
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Score Médio (COM)", className="card-title"),
            html.H2(f"{avg_score_com:.2f}", className="card-text")
        ]), color="primary", outline=True)), 
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Score Médio (SEM)", className="card-title"),
            html.H2(f"{avg_score_sem:.2f}", className="card-text")
        ]), color="secondary", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Reviews Médios (COM)", className="card-title"),
            html.H2(f"{avg_reviews_com:,.0f}", className="card-text")
        ]), color="primary", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Reviews Médios (SEM)", className="card-title"),
            html.H2(f"{avg_reviews_sem:,.0f}", className="card-text")
        ]), color="secondary", outline=True)),
    ], className="mb-4"),

    # Linha 2: Gráficos (lado a lado)
    dbc.Row([
        dbc.Col(dcc.Graph(id='grafico-violino', figure=fig_violino), width=6),
        dbc.Col(dcc.Graph(id='grafico-ecdf', figure=fig_ecdf), width=6),
    ]),
    # Linha 2: Gráficos (lado a lado)
dbc.Row([
    dbc.Col(dcc.Graph(id='grafico-violino', figure=fig_violino), width=6),
    dbc.Col(dcc.Graph(id='grafico-ecdf', figure=fig_ecdf), width=6),
]), # <--- ADICIONE ESTA VÍRGULA

# ******************************************************
# --- INÍCIO - PASSO 2 (Adicionar Tabelas ao Layout) ---

dbc.Row([
    # Tabela de 'Reviews'
    dbc.Col([
        html.H5("Estatísticas Descritivas (Reviews)", className="mt-4"),
        dash_table.DataTable(
            data=df_desc_reviews.to_dict('records'),
            columns=[
                {'name': 'Métrica', 'id': 'Métrica'},
                {'name': 'SEM Website', 'id': 'Reviews sem site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')},
                {'name': 'COM Website', 'id': 'Reviews com site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')}
            ],
            style_table={'overflowX': 'auto', 'height': '450px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Arial, sans-serif'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': '1px solid #dee2e6'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
        )
    ], width=6),

    # Tabela de 'TotalScore'
    dbc.Col([
        html.H5("Estatísticas Descritivas (TotalScore)", className="mt-4"),
        dash_table.DataTable(
            data=df_desc_score.to_dict('records'),
            columns=[
                {'name': 'Métrica', 'id': 'Métrica'},
                {'name': 'SEM Website', 'id': 'TotalScore sem site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')},
                {'name': 'COM Website', 'id': 'TotalScore com site', 'type': 'numeric', 'format': Format(precision=2, scheme='f')}
            ],
            style_table={'overflowX': 'auto', 'height': '450px', 'overflowY': 'auto'},
            style_cell={'textAlign': 'left', 'fontFamily': 'Arial, sans-serif'},
            style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': '1px solid #dee2e6'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
        )
    ], width=6)
], className="mb-4")

# --- FIM - PASSO 2 ---
# ******************************************************

]) # <--- Fim do html.Div
# --- Fim do Layout da Aba 1 ---

# --- Layout da Aba 2 (Regressão) ---
layout_regressao = html.Div([
    # Linha 0: Texto da Conclusão
    dcc.Markdown("""
        ##### Ato 2: A Falha da Regressão
        Como exigido, testamos exaustivamente a hipótese de que a Popularidade (`reviewsCount`)
        poderia prever a Qualidade (`totalScore`). A conclusão unânime de todos os métodos foi:
        
        **A hipótese falhou. O R² de ~0.0003 prova que as variáveis são independentes.**
    """, className="mb-4"),
    
    # Linha 1: A Prova Visual
    dbc.Row([
        dbc.Col(dcc.Graph(id='grafico-regressao-r2', figure=fig_regressao_r2), width=6),
        
        # A solução final
        # Trocamos o placeholder pela imagem real da pasta /assets
        dbc.Col(
            html.Img(
            # Assumindo que o nome é este
            src=app.get_asset_url('modelos.png'),
            # Estilo para garantir que a imagem se ajuste bem
            style={'height': '100%', 'width': '100%', 'max-height': '450px', 'object-fit': 'contain'}
        ), 
    width=6
),

    ], className="mb-4"),
    
    # Linha 2: A Prova Numérica
    dbc.Row(
        dbc.Col(
            html.Div([
                html.H5("Tabela de Métricas Detalhadas (Regressão)"),
                dash_table.DataTable(
                    data=df_regressao_metricas.to_dict('records'),
                    # Adicionando formatação de números
                    columns=[
                        {'name': 'Modelo', 'id': 'Modelo'},
                        {'name': 'R²', 'id': 'R2_Valores', 'type': 'numeric', 
                        'format': Format(precision=6, scheme='f')},
                        {'name': 'RMSE', 'id': 'RMSE', 'type': 'numeric', 
                        'format': Format(precision=6, scheme='f')}
                    ],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'fontFamily': 'Arial, sans-serif'},
                    style_header={
                        'backgroundColor': '#f8f9fa', # Cor suave do Yeti
                        'fontWeight': 'bold',
                        'border': '1px solid #dee2e6'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)',
                        }
                    ],
                    style_cell_conditional=[
                        {'if': {'column_id': 'Modelo'},
                         'width': '30%'},
                    ]
                )
            ])
        )
    )
])
# --- Fim do Layout da Aba 2 ---

# --- Fim do Layout da Aba 2 ---

# ******************************************************
# --- INÍCIO DO NOVO CÓDIGO (PATCH 3) ---
# --- Layout da Aba 3 (Machine Learning) ---

# Layout da Sub-Aba 3.1: Clustering
layout_ml_cluster = html.Div([
    dcc.Markdown("#### 1. Aprendizado Não-Supervisionado (Clustering)"),
    dcc.Markdown("Exploração da estrutura dos dados (`totalScore` vs `reviewsCount`)."),
    
    dbc.Row([
        # Controlo
        dbc.Col(
            dbc.Card([
                html.H5("Selecione o Método", className="card-title"),
                dcc.Dropdown(
                    id='dropdown-cluster',
                    options=[
                        {'label': 'K-Means', 'value': 'K-Means'},
                        {'label': 'Hierárquico', 'value': 'Hierárquico'},
                        {'label': 'EM (GMM)', 'value': 'EM (GMM)'},
                        {'label': 'DBSCAN', 'value': 'DBSCAN'}
                    ],
                    value='K-Means' # Valor inicial
                ),
                html.H5("Métricas de Avaliação", className="card-title mt-4"),
                dbc.Row([
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Silhouette Score"),
                        html.H4(id='card-silhouette', children="0.00")
                    ]), color="primary", outline=True)),
                    dbc.Col(dbc.Card(dbc.CardBody([
                        html.H6("Davies-Bouldin"),
                        html.H4(id='card-davies', children="0.00")
                    ]), color="secondary", outline=True)),
                ])
            ]), 
        width=4),
        
        # Gráfico
        # Gráfico (agora um 'contentor' para a nossa imagem)
        dbc.Col(
            html.Div(id='cluster-graph-container'), # <-- MUDANÇA (novo ID)
                width=8)
        ])
])

# Layout da Sub-Aba 3.2: Classificação
layout_ml_classif = html.Div([
    dcc.Markdown("#### 2. Aprendizado Supervisionado (Classificação)"),
    dcc.Markdown("Treinamento de modelos para prever `has_website` usando `totalScore` e `reviewsCount`."),
    
    # Controlo
    dbc.Row(
        dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H5("Selecione o Modelo", className="card-title"),
                dcc.Dropdown(
                    id='dropdown-classif',
                    options=[
                        {'label': 'Árvore de Decisão', 'value': 'Árvore de Decisão'},
                        {'label': 'Random Forest', 'value': 'Random Forest'},
                        {'label': 'KNN', 'value': 'KNN'},
                        {'label': 'Rede Neural (MLP)', 'value': 'Rede Neural (MLP)'}
                    ],
                    value='Random Forest' # Valor inicial
                ),
            ])),
        width=12), 
    className="mb-3"),
    
    # Métricas
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Acurácia"), html.H4(id='card-accuracy')]), color="success", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("F1-Score"), html.H4(id='card-f1')]), color="success", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Precision"), html.H4(id='card-precision')]), color="primary", outline=True)),
        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Recall"), html.H4(id='card-recall')]), color="primary", outline=True)),
    ], className="mb-3"),
    
    # Gráficos
    # Gráficos
    dbc.Row([
        dbc.Col(
            html.Div(id='classif-graph-container'), # <-- UM contentor, com um novo ID
            width=12
        ) 
    ])
])

# Layout da Sub-Aba 3.3: Comparativo
layout_ml_comparativo = html.Div([
    dcc.Markdown("#### 3. Comparativo Final (Classificação)"),
    dcc.Markdown("Qual modelo teve o melhor desempenho na tarefa de classificação?"),
    dbc.Row(dbc.Col(
        dcc.Graph(id='graph-classif-comparativo', figure=fig_classif_comparativo), 
    width=12)),
    
    dbc.Row(dbc.Col(
        html.Div([
            html.H5("Tabela de Métricas Detalhadas (Classificação)"),
            dash_table.DataTable(
                data=df_classif_comparativo.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_classif_comparativo.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'fontFamily': 'Arial, sans-serif'},
                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'border': '1px solid #dee2e6'},
            )
        ]),
    className="mt-4"))
])


# Layout Principal da Aba 3 (contém as sub-abas)
layout_ml = html.Div([
    dcc.Markdown("""
        ##### Ato 3: A Solução
        Tendo provado que a Regressão (X → Y) falhou, pivotamos o problema. 
        Usamos **Clustering** para explorar os dados e **Classificação** para prever `has_website` a partir de `totalScore` e `reviewsCount`.
    """, className="mb-4"),
    
    dbc.Tabs(id='tabs-ml', children=[
        dbc.Tab(layout_ml_cluster, label='1. Não-Supervisionado (Clustering)'),
        dbc.Tab(layout_ml_classif, label='2. Supervisionado (Classificação)'),
        dbc.Tab(layout_ml_comparativo, label='3. Comparativo Final'),
    ])
])

# Define o layout principal
app.layout = dbc.Container([
    # Título do Dashboard
    html.H1("Ciência de Dados em Hotéis", 
            className="text-center my-4",
            style={'color': '#005A9C'}), 
    
    # Sistema de abas
    dcc.Tabs(id='tabs-principal', value='tab-descritiva', children=[
        dcc.Tab(label='Análise Descritiva', value='tab-descritiva'),
        dcc.Tab(label='Regressão', value='tab-regressao'), 
        dcc.Tab(label='Machine Learning', value='tab-ml'),
    ], className="mb-3"),
    
    # Container onde o conteúdo das abas será exibido
    html.Div(id='conteudo-tabs', className='my-4')
    
], fluid=True, style={'backgroundColor': '#FFFFFF'}) 


# Callback para alternar o conteúdo das abas
@app.callback(
    Output('conteudo-tabs', 'children'),
    Input('tabs-principal', 'value')
)
def renderizar_conteudo(aba_selecionada):
    if aba_selecionada == 'tab-descritiva':
        return layout_descritiva
    
    elif aba_selecionada == 'tab-regressao': 
        return layout_regressao # <--- Agora preenchido
    
    elif aba_selecionada == 'tab-ml':
        return layout_ml # <--- Retorna o layout complexo da Aba 3

# --- INÍCIO DO NOVO CÓDIGO (PATCH 5) ---
# --- Callbacks da Aba 3 (Machine Learning) ---

@app.callback(
    [Output('cluster-graph-container', 'children'), # <-- MUDANÇA (novo ID e 'children')
     Output('card-silhouette', 'children'),
     Output('card-davies', 'children')],
    [Input('dropdown-cluster', 'value')]
)
def update_cluster_visuals(metodo_selecionado):
    # Procura os dados
    data = cluster_data.get(metodo_selecionado, cluster_data['K-Means'])

    # --- LÓGICA NOVA PARA IMAGEM (MUITO MAIS SIMPLES) ---

    # 1. Pega o nome do ficheiro de imagem
    img_filename = data.get('img_filename', 'default.png') # Pega o nome

    # 2. Cria o componente de Imagem HTML
    img_component = html.Img(
        src=app.get_asset_url(img_filename), # Puxa da pasta /assets
        # Estilo para garantir que a imagem ocupe o espaço (corrigindo o 'pequeno')
        style={'width': '100%', 'height': '100%', 'object-fit': 'contain'}
    )

    # --- FIM DA LÓGICA NOVA ---

    # Formata Silhouette (lógica antiga)
    sil_score = f"{data['silhouette']:.3f}"

    # Formata Davies-Bouldin (lógica antiga)
    db_value = data['davies_bouldin']
    if db_value is None:
        dav_score = "N/A"
    else:
        dav_score = f"{db_value:.3f}"

    # Retorna o componente de Imagem, e não mais um 'figure'
    return img_component, sil_score, dav_score

# Callback para a Sub-Aba 3.2: Classificação
@app.callback(
    [Output('card-accuracy', 'children'),
     Output('card-f1', 'children'),
     Output('card-precision', 'children'),
     Output('card-recall', 'children'),
     Output('classif-graph-container', 'children')], # <-- MUDANÇA (só 1 gráfico)
    [Input('dropdown-classif', 'value')]
)
def update_classification_visuals(modelo_selecionado):
    # Procura os dados (mock data)
    data = classif_data.get(modelo_selecionado, classif_data['Random Forest'])

    # Formata as métricas
    accuracy = f"{data['accuracy']:.2%}"
    f1 = f"{data['f1']:.2%}"
    precision = f"{data['precision']:.2%}"
    recall = f"{data['recall']:.2%}"

    # --- LÓGICA DE IMAGEM SIMPLIFICADA ---

    # Pega o nome do ficheiro de imagem
    img_filename = data.get('img_filename', 'default.png')

    # Cria o componente de Imagem
    img_component = html.Img(
        src=app.get_asset_url(img_filename),
        style={'width': '100%', 'height': '100%', 'object-fit': 'contain'}
    )

    # --- FIM DA LÓGICA ---

    return accuracy, f1, precision, recall, img_component

# Executa o servidor
if __name__ == '__main__':
    app.run(debug=True, port=8050)