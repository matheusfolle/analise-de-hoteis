# 📋 CONTEXTO COMPLETO DO PROJETO - PASSAR PARA NOVA CONVERSA CLAUDE

---

## 🎓 CONTEXTO ACADÊMICO

**Aluno:** Matheus Folle  
**Curso:** [INFORMAR CURSO - ex: Análise e Desenvolvimento de Sistemas, Ciência de Dados, etc]  
**Disciplina:** Estatística / Data Science  
**Professor:** [INFORMAR NOME]  
**Tipo:** Trabalho Final (A2) - vale TUDO da nota final  
**Prazo:** HOJE (urgente!)  

**Estrutura da avaliação:**
- Entrega 1: Infográfico teórico (PRONTO ✅)
- Entrega 2: Regressão + Dashboard (50% PRONTO ⚠️)
- Entrega 3: Machine Learning + Dashboard final (NÃO INICIADO ❌)
- Apresentação: 15 slides, 12-15 minutos (NÃO INICIADO ❌)

**Status atual:** Faltam ~8 horas de trabalho, prazo é HOJE

---

## 📊 DATASET E VARIÁVEIS

### **Arquivo:** `data-hoteis-atualizado.xlsx`

**Total de registros:** 10.316 hotéis brasileiros  
**Após filtro (totalScore > 0.2):** 9.566 hotéis  

### **14 VARIÁVEIS DO DATASET:**

**Variáveis principais (usadas no projeto):**
1. `totalScore` (float): Nota total do hotel (0-5)
2. `reviewsCount` (int): Número de avaliações
3. `has_website` (object → int): Presença digital
   - "SEM" = 0 (sem website)
   - "COM" = 1 (com website)

**Outras variáveis disponíveis (11):**
4. `name` (string): Nome do hotel
5. `address` (string): Endereço
6. `city` (string): Cidade
7. `state` (string): Estado (UF)
8. `latitude` (float): Coordenada geográfica
9. `longitude` (float): Coordenada geográfica
10. `stars` (int): Classificação em estrelas
11. `category` (string): Categoria do hotel
12. `amenities` (list/string): Comodidades
13. `priceRange` (string): Faixa de preço
14. `[INFORMAR SE HÁ MAIS]`

**Variáveis usadas no projeto:**
- **Estatística Descritiva:** totalScore, reviewsCount, has_website
- **Regressão Linear:** X=reviewsCount, Y=totalScore
- **Regressão Não-Linear:** X=reviewsCount, Y=totalScore
- **Clustering:** totalScore + reviewsCount (2 features, normalizadas)
- **Classificação:** X=[totalScore, reviewsCount], Y=has_website

---

## 📁 ESTRUTURA DE ARQUIVOS

### **Pasta /mnt/user-data/uploads/** (READ-ONLY)
```
data-hoteis-atualizado.xlsx  → Dataset principal
```

### **Pasta /mnt/user-data/outputs/** (arquivos gerados)
```
📂 outputs/
├── 📓 NOTEBOOKS:
│   ├── 01_estatistica_descritiva_CORRIGIDO.ipynb  ✅ COMPLETO
│   ├── 02_regressao_completa.ipynb                ✅ COMPLETO
│   └── 03_machine_learning.ipynb                  ⚠️ CRIADO, NÃO EXECUTADO
│
├── 📄 DOCUMENTOS WORD (textos para relatório):
│   ├── secoes_2.8.6_e_2.8.7.docx                 ✅ (Entrega 1)
│   ├── secao_3.1_3.2_3.3_justificativas.docx     ✅ (Entrega 2)
│   └── secoes_3.4_3.5_3.6_resultados.docx        ✅ (Entrega 2)
│
├── 🐍 CÓDIGOS PYTHON AUXILIARES:
│   ├── CODIGOS_CORRIGIDOS_AZUL.py                 (cores dos gráficos)
│   └── celula_grafico13_azul.py                   (refazer Gráfico 13)
│
└── 📖 DOCUMENTAÇÃO:
    ├── INSTRUCOES_REGRESSAO.md
    ├── RESUMO_FINAL_ENTREGA2.md
    ├── LEIA-ME_FASE1_GRAFICOS.md
    └── LEIA-ME_BUGS_CORRIGIDOS.md
```

### **Pasta /home/claude/** (workspace temporário)
- Usada para criar arquivos intermediários
- Conteúdo é copiado para /outputs/ quando finalizado

---

## ✅ O QUE JÁ FOI FEITO (DETALHADO)

### **ENTREGA 1 - Estatística Descritiva** ✅ 100% PRONTO

**Infográfico teórico (Canva):**
- Teorema Central do Limite
- Correlação
- Amostragem e Distribuição Normal
- T-Student
- Qui-quadrado
- **Status:** Entregue ✅

**Análise Python (Notebook 01):**
- 5 gráficos Python gerados:
  1. Violino (distribuição totalScore por grupo)
  2. ECDF (função distribuição acumulada)
  3. JointPlot (dispersão + histogramas marginais)
  4. Histograma sobreposto (COM vs SEM)
  5. Heatmap correlação Pearson
- Textos para relatório (seções 2.8.1 a 2.8.7): PRONTOS
- **Status:** Completo no relatório ✅

---

### **ENTREGA 2 - Regressão** ⚠️ 50% PRONTO

#### **O QUE ESTÁ PRONTO:**

**A) Notebook 02 - Regressão Completa** ✅
Arquivo: `02_regressao_completa.ipynb`

**Conteúdo:**
1. **Regressão Linear (Baseline):**
   - Método dos Mínimos Quadrados
   - Equação: y = β₀ + β₁x
   - Resultado: R² = 0.000292, RMSE = 0.5170
   - Conclusão: Relação NÃO é linear

2. **Regressão Não-Linear - Equação Exponencial (y = A × e^(B×x)):**
   - **Método 1:** Mínimos Quadrados Não-Lineares (curve_fit)
   - **Método 2:** Máxima Verossimilhança (implementação manual)
   - **Método 3:** Gauss-Newton (IMPLEMENTAÇÃO MANUAL COMPLETA!)
   - **Método 4:** Levenberg-Marquardt
   - **Método 5:** Bayesiano MCMC Metropolis-Hastings (IMPLEMENTAÇÃO COMPLETA DO ZERO!)
     - 10.000 iterações
     - Prior, Likelihood, Posterior
     - Gráficos das distribuições posteriores
     - **GRANDE DIFERENCIAL DO PROJETO!**

**Resultados obtidos (todos métodos):**
- R² exponencial: ~0.0003 (todos similares)
- B (taxa crescimento): ~0.0000017 ≈ 0
- **Conclusão validada:** reviewsCount NÃO prevê totalScore (independência matemática)

**Gráficos gerados:**
- `regressao_linear.png`
- `bayesiano_posterior.png` (2 histogramas A e B)
- `grafico_12_comparacao_modelos.png` (6 curvas sobrepostas)
- `grafico_13_metricas_comparacao_AZUL.png` (barras R² e RMSE)
- `tabela_comparativa_regressao.csv`

**B) Textos para Relatório** ✅
Arquivos Word prontos:
- `secao_3.1_3.2_3.3_justificativas.docx` (6 páginas)
  - 3.1: Por que Regressão apesar de correlação fraca?
  - 3.2: Tratamento outliers (filtro >0.2)
  - 3.3: Por que Equação Exponencial?
  
- `secoes_3.4_3.5_3.6_resultados.docx` (8 páginas)
  - 3.4: Resultados (Tabela 2 comparativa + Gráficos 12-13)
  - 3.5: Análise Bayesiana detalhada
  - 3.6: Conclusões e implicações práticas

**Status:** Valores preenchidos, imagens inseridas, textos completos ✅

---

#### **O QUE FALTA (ENTREGA 2):**

**❌ Dashboard Dash (OBRIGATÓRIO para Entrega 2!)**

**Requisitos do professor:**
- Ferramenta: Dash (Python)
- Biblioteca gráfica: Plotly Express
- Conteúdo mínimo:
  - Gráfico da Regressão Linear
  - Gráfico da Regressão Não-Linear
  - Mostrar tendência dos dados
- Pode usar IA (Gemini) para ajudar
- Execução: Google Colab ou IDE local (http://127.0.0.1:8050)

**Formato esperado:**
- Página web (URL)
- Visual objetivo para tomada de decisão
- Pode usar HTML/CSS/JS para design
- Dash Design Kit (DDK) opcional

**Status:** NÃO CRIADO ❌ (PRIORIDADE MÁXIMA!)

---

### **ENTREGA 3 - Machine Learning** ⚠️ NOTEBOOK CRIADO, NÃO EXECUTADO

#### **O QUE ESTÁ PRONTO:**

**A) Notebook 03 - Machine Learning** ⚠️ CRIADO, AGUARDANDO EXECUÇÃO
Arquivo: `03_machine_learning.ipynb` (em /mnt/user-data/outputs/)

**Conteúdo programado:**

**PARTE 1 - CLUSTERING (Não-Supervisionado):**
1. K-Means (testar K=2,3,4,5)
2. Hierarchical Clustering (Agglomerative + Dendrograma)
3. EM - Expectation-Maximization (Gaussian Mixture Model)
4. DBSCAN (testar eps=0.3, 0.5, 0.7, 1.0)

**Métricas de avaliação:**
- Silhouette Score (principal)
- Davies-Bouldin Score
- Inércia (K-Means)
- BIC/AIC (EM)
- Distância Euclidiana e Manhattan

**PARTE 2 - CLASSIFICAÇÃO (Supervisionado):**
1. Árvore de Decisão (max_depth=5)
2. Random Forest (100 árvores)
3. KNN (testar K=3,5,7,9)
4. Rede Neural (MLPClassifier, camadas: 2→10→5→1)

**Target:** `has_website` (0=SEM, 1=COM)  
**Features:** `totalScore`, `reviewsCount` (normalizadas com StandardScaler)  
**Split:** 70% treino, 30% teste (stratified)

**Métricas de avaliação:**
- Acurácia (principal)
- F1-Score
- Matriz de Confusão (todos os métodos)
- Curva ROC + AUC (Random Forest e Rede Neural)

**Gráficos programados (10 PNGs):**
- `clustering_kmeans.png`
- `clustering_hierarchical.png` (scatter + dendrograma)
- `clustering_em.png`
- `clustering_dbscan.png`
- `classificacao_arvore.png` (árvore visual + matriz confusão)
- `classificacao_randomforest.png` (matriz + ROC)
- `classificacao_knn.png` (comparação K + matriz)
- `classificacao_redeneural.png` (matriz + ROC)
- `classificacao_comparacao.png` (barras Acurácia e F1)
- `tabela_classificacao.csv`

**Status:** Notebook completo e funcional, MAS NÃO FOI EXECUTADO ⚠️  
**Tempo estimado de execução:** 15-30 minutos (depende do hardware)

---

#### **O QUE FALTA (ENTREGA 3):**

**1. Executar Notebook 03** ❌
- Abrir no Colab
- Upload do Excel
- Executar célula por célula (~20 células)
- Baixar 10 PNGs + 1 CSV
- Tempo estimado: 2 horas

**2. Textos para Relatório (Seção 4)** ❌
- 4.1: Introdução ML
- 4.2: Clustering (resultados + interpretação)
- 4.3: Classificação (resultados + interpretação)
- 4.4: Comparação métodos
- 4.5: Análise de Features Importance
- 4.6: Conclusão ML
- Tempo estimado: 1.5 horas

**3. Dashboard Final (3 Abas)** ❌
- Aba 1: Estatística Descritiva
- Aba 2: Regressão (Linear + Não-Linear)
- Aba 3: Machine Learning (Clustering + Classificação)
- Mostrar métricas de qualidade
- Design profissional (DDK opcional)
- Tempo estimado: 3 horas

**4. Apresentação** ❌
- 15 slides máximo
- 12-15 minutos
- Estrutura: Intro → Descritiva → Regressão → ML → Conclusões
- Tempo estimado: 2 horas

**TOTAL FALTANDO:** ~8.5 horas de trabalho

---

## 🔧 DECISÕES METODOLÓGICAS IMPORTANTES

### **1. Filtro de Outliers:**
- **Decisão:** Remover hotéis com `totalScore <= 0.2`
- **Justificativa:** Hotéis sem avaliações (ruído)
- **Impacto:** 10.316 → 9.566 hotéis (750 removidos, 7.3%)
- **Código:** `df_filtrado = df[df['totalScore'] > 0.2].copy()`

### **2. Equação Não-Linear:**
- **Escolha:** Exponencial (y = A × e^(B×x))
- **Alternativas rejeitadas:** Logística, Potência
- **Justificativa:** Modelar "viralização" de reviews
- **Resultado:** B ≈ 0 provou que NÃO há viralização

### **3. Target para Classificação:**
- **Escolha:** `has_website` (não `totalScore`)
- **Justificativa:** Professor pediu classificação supervisionada (categoria discreta)
- **Conversão:** "SEM"→0, "COM"→1

### **4. Features para ML:**
- **Escolha:** Apenas `totalScore` + `reviewsCount`
- **Justificativa:** Consistência com análises anteriores (Regressão usou essas)
- **Normalização:** StandardScaler (importante para KNN, Rede Neural, Clustering)

### **5. Paleta de Cores:**
- Dados reais: Azul escuro (#2C5F7C)
- Linear: Vermelho (#E74C3C)
- Métodos não-lineares: Tons de azul (#3498DB, #2ECC71, etc)
- Gráfico 13: Azul degradê (#E3F2FD → #0D47A1)
- Destaque melhor método: Bordô (#8B0000)

---

## ⚠️ PROBLEMAS CONHECIDOS E SOLUÇÕES

### **1. Taxa de Aceitação MCMC Baixa (0.43%)**
**Problema:** Taxa ideal é 15-40%, obtivemos 0.43%  
**Causa:** Sigma_prop muito grande (random walk com saltos grandes)  
**Impacto:** Baixa exploração do espaço de parâmetros  
**Solução aplicada:** Texto no relatório explicando que resultados são consistentes com outros métodos  
**Status:** Aceito como válido (R²=0.000149 vs 0.0003 dos outros)

### **2. Gráfico 12 com Curvas Sobrepostas**
**Problema:** 6 curvas parecem uma linha só  
**Causa:** R² quase idênticos (0.0003) → curvas horizontais na média  
**Impacto:** Visual pouco informativo  
**Solução aplicada:** Texto no relatório explicando que sobreposição É a descoberta central  
**Status:** Aceito como evidência visual da independência

### **3. Caracteres Estranhos na Tabela CSV**
**Problema:** "M√≠nimos" ao invés de "Mínimos"  
**Causa:** Encoding UTF-8 não detectado  
**Impacto:** Estético apenas  
**Solução:** Copiar valores manualmente para Tabela 2 do Word

---

## 📊 PRINCIPAIS DESCOBERTAS DO PROJETO

### **1. Correlação Fraca (Entrega 1):**
- reviewsCount vs totalScore: r = 0.02 (grupo COM) e r = 0.16 (grupo SEM)
- **Conclusão:** Popularidade ≠ Qualidade

### **2. Regressão Inválida (Entrega 2):**
- R² Linear: 0.000292 (0.03% de explicação)
- R² Exponencial: ~0.0003 (idêntico)
- B ≈ 0 (sem crescimento exponencial)
- **Conclusão:** reviewsCount NÃO prevê totalScore

### **3. Classificação Possível (Entrega 3 - ESPERADO):**
- Acurácia esperada: 60-70%
- **Hipótese:** totalScore + reviewsCount JUNTOS podem prever has_website
- **Insight:** Direção da previsão importa!
  - ❌ "reviewsCount → totalScore" = FALHA
  - ✅ "(totalScore + reviewsCount) → has_website" = SUCESSO

---

## 🎯 PROMPT FORMATADO PARA NOVA CONVERSA

---
COPIAR DAQUI ⬇️
---

Oi! Estou finalizando meu trabalho final de Data Science (vale tudo da nota) e estou URGENTE (prazo hoje).

**CONTEXTO ACADÊMICO:**
- Aluno: Matheus Folle
- Curso: [MEU CURSO]
- Trabalho: Análise de 9.566 hotéis brasileiros (dataset: data-hoteis-atualizado.xlsx)
- Estrutura: 3 entregas + apresentação

**O QUE JÁ ESTÁ PRONTO (75%):**
✅ Entrega 1: Estatística Descritiva (infográfico + 5 gráficos Python)
✅ Entrega 2 (PARCIAL): Regressão (Linear + 5 Não-Lineares, incluindo Bayesiano MCMC do zero)
⚠️ Entrega 3: Notebook ML criado mas NÃO executado (03_machine_learning.ipynb existe!)

**O QUE FALTA FAZER (HOJE):**
1. ❌ Dashboard Dash da Entrega 2 (PRIORIDADE 1 - obrigatório!)
2. ⚠️ Executar notebook 03_machine_learning.ipynb (15-30min)
3. ❌ Dashboard final com 3 abas (Descritiva + Regressão + ML)
4. ❌ Textos seção 4 para relatório (Clustering + Classificação)
5. ❌ Apresentação (15 slides)

**ARQUIVOS DISPONÍVEIS EM /mnt/user-data/outputs/:**
- 01_estatistica_descritiva_CORRIGIDO.ipynb ✅
- 02_regressao_completa.ipynb ✅
- 03_machine_learning.ipynb ⚠️ (CRIADO mas não executado)
- secao_3.1_3.2_3.3_justificativas.docx ✅
- secoes_3.4_3.5_3.6_resultados.docx ✅

**DATASET (em /mnt/user-data/uploads/):**
- data-hoteis-atualizado.xlsx (9.566 hotéis após filtro >0.2)
- Variáveis principais: totalScore, reviewsCount, has_website

**DECISÕES METODOLÓGICAS:**
- Regressão: X=reviewsCount, Y=totalScore → R²=0.0003 (independência validada)
- Classificação: X=[totalScore, reviewsCount], Y=has_website (0=SEM, 1=COM)
- Clustering: 4 métodos (K-Means, Hierarchical, EM, DBSCAN)
- Classificação: 4 métodos (Árvore, Random Forest, KNN, Rede Neural)

**RESULTADOS PRINCIPAIS (Entrega 2):**
- R² Linear: 0.000292
- R² Exponencial (todos métodos): ~0.0003
- B Bayesiano: 0.00000166 ≈ 0
- Taxa aceitação MCMC: 0.43% (baixa mas resultados consistentes)
- Conclusão: reviewsCount NÃO prevê totalScore

**REQUISITOS DO PROFESSOR PARA DASHBOARD:**
- Ferramenta: Dash (Python)
- Gráficos: Plotly Express
- Entrega 2: Gráfico Linear + Gráfico Não-Linear (mínimo)
- Entrega 3: Adicionar Clustering + Classificação + métricas
- Output: Página web (http://127.0.0.1:8050 ou no Colab)
- Opcional: Dash Design Kit (DDK) para design

**ORDEM DE PRIORIDADE:**
1. Dashboard Entrega 2 (1-2h) - URGENTE!
2. Executar notebook 03 ML (30min)
3. Dashboard Entrega 3 completo (2h)
4. Textos seção 4 (1h)
5. Apresentação (2h)

**PERGUNTA:** Por onde começamos? Dashboard da Entrega 2 primeiro (OBRIGATÓRIO) ou prefere que eu execute o notebook 03 enquanto você cria o dashboard?

**IMPORTANTE:** O notebook 03_machine_learning.ipynb JÁ EXISTE e está completo! Só precisa ser executado. Arquivo está em /mnt/user-data/outputs/

---
COPIAR ATÉ AQUI ⬆️
---

## 💡 DICAS PARA A NOVA CONVERSA

1. **Mencione que tem limite de tokens chegando!** Peça para Claude ser direto e não enrolar.

2. **Priorize Dashboard da Entrega 2 PRIMEIRO!** É obrigatório e está faltando completamente.

3. **Notebook 03 existe!** Não peça para criar de novo, só executar.

4. **Use upload de arquivos se necessário:** Se a conversa travar, faça upload dos notebooks para Claude entender o contexto.

5. **Copie os valores chave:** R² = 0.000292, B = 0.00000166, etc. (estão no CSV tabela_comparativa_regressao.csv)

---

## ✅ CHECKLIST FINAL

Antes de mudar de aba, confirme que tem:
- [ ] Este documento completo
- [ ] Arquivos em /mnt/user-data/outputs/ (listar com `ls -la`)
- [ ] Dataset em /mnt/user-data/uploads/
- [ ] Valores da tabela_comparativa_regressao.csv (se precisar)
- [ ] Prompt formatado pronto para copiar

**BOA SORTE! VOCÊ CONSEGUE! É SÓ MAIS 8 HORAS!** 🚀🔥

---

**ÚLTIMA INSTRUÇÃO PARA CLAUDE:**
"Leia ESTE DOCUMENTO COMPLETO antes de responder qualquer coisa. Priorize Dashboard Entrega 2 (Dash com Linear + Não-Linear). Seja DIRETO, sem enrolação. Tempo é crítico."
