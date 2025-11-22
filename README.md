<div align="center">

# Impacto da Presença Digital em Hotéis Brasileiros
### *Pipeline de Análise Estatística e Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Colab-Ready-orange.svg)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

**Análise de 10.316 hotéis brasileiros revelando a relação entre presença digital, engajamento e qualidade**

</div>

---

## 📖 Sobre o Projeto

Este projeto de Data Science analisa a complexa relação entre:
- **Presença Digital** (`has_website`)
- **Engajamento/Popularidade** (`reviewsCount`)
- **Performance/Qualidade** (`totalScore`)

Através de uma pipeline rigorosa de 3 fases, descobrimos insights surpreendentes sobre o setor hoteleiro brasileiro.

---

## 🎯 Questões-Chave

1. Ter um website correlaciona com mais avaliações e notas superiores?
2. A popularidade de um hotel (nº de reviews) pode prever sua qualidade (score)?
3. Inversamente, o perfil de performance pode prever a estratégia digital?

---

## 🔬 A Pipeline de Análise (3 Fases)

### **Fase 1: Estatística Descritiva**
Análise exploratória confirmou que presença digital está fortemente associada a performance superior:

| Métrica | Hotéis COM Website | Hotéis SEM Website | Diferença |
|---------|-------------------|-------------------|-----------|
| **Reviews Médios** | 779 | 90 | **+767%** |
| **Score Médio** | 4.38 | 4.26 | **+2.8%** |

**Conclusão:** Presença digital importa (p < 0.001).

---

### **Fase 2: Análise de Regressão** *(Uma Descoberta no "Fracasso")*

Testamos rigorosamente se popularidade (`reviewsCount`) poderia prever qualidade (`totalScore`).

**6 Modelos Testados:**
- Regressão Linear
- 5 Variações Não-Lineares Exponenciais
- MCMC Bayesiano

**Resultado:** Todos falharam (R² ≈ 0.0003)

> **Descoberta Científica:** As variáveis são estatisticamente independentes. 
> Um hotel pode ser popular e ruim, ou exclusivo e excelente.

---

### **Fase 3: Machine Learning** *(A "Virada" Metodológica)*

Dado que X → Y falhou, reformulamos: **(X + Y) → Z**

Usamos `totalScore` + `reviewsCount` como features para prever `has_website`.

**4 Algoritmos de Clustering:**
- K-Means
- Hierarchical Clustering
- Expectation Maximization (EM)
- DBSCAN

**4 Algoritmos de Classificação:**
- Decision Tree (Vencedor)
- Random Forest
- K-Nearest Neighbors (KNN)
- Neural Network (MLP)

---

## 🏆 Resultado Final
```
Modelo Vencedor: Decision Tree
Acurácia: 76,30%
F1-Score: 0.77
```

**Conclusão:** Embora popularidade e qualidade sejam independentes, o **perfil de performance** de um hotel (score + reviews) é um forte preditor de sua **estratégia de presença digital**.

---

## 📁 Estrutura do Projeto
```
📦 hotels-analysis/
├── 📓 notebooks/
│   ├── 01_estatistica_descritiva.ipynb
│   ├── 02_regressao_completa.ipynb
│   └── 03_machine_learning.ipynb
├── 📊 data/
│   ├── raw/
│   │   └── data-hoteis-atualizado.xlsx
│   └── cleaned/
│       └── data-hoteis.xlsx
├── 📄 papers/
│   ├── report.docx
│   └── infographic.pdf
├── 📈 dashboard/
│   ├── app.py
│   └── requirements.txt
├── 🐍 src/
│   └── analysis.py
└── 📖 README.md
```

---

## Como Executar

### Pré-requisitos
```bash
Python 3.8+
Google Colab
```

### Instalação
```bash
# Clone o repositório
git clone https://github.com/matheusfolle/hotels-analysis.git

# Entre no diretório
cd hotels-analysis

# Instale as dependências
pip install -r dashboard/requirements.txt
```

### Executar Notebooks no Google Colab

1. **`01_estatistica_descritiva.ipynb`** - EDA e testes estatísticos
2. **`02_regressao_completa.ipynb`** - Modelagem preditiva (6 modelos)
3. **`03_machine_learning.ipynb`** - Clustering e Classificação

### Executar Dashboard Interativo
```bash
cd dashboard
python app.py
```

O dashboard estará disponível em `http://localhost:8050`

---

## 💻 Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

---

## 📊 Infográfico - Fundamentos Estatísticos

Este projeto aplica diversos conceitos estatísticos fundamentais.

**Conceitos abordados:**
- Teorema Central do Limite
- Correlação de Pearson
- Amostragem e Distribuição Normal
- Teste T-Student
- Teste Qui-Quadrado

- [Ver Infográfico Completo no Google Drive](https://drive.google.com/file/d/1ilGq6ronvuChRIt-kg_YC1RSQGcE8HJN/view?usp=drive_link)

---

## 📚 Materiais do Projeto

- **Paper Completo:** 
  - [📥 Download (GitHub)](papers/report.docx)
  - [👁️ Visualizar Online (Google Drive)](https://docs.google.com/document/d/1yF0LLgoXG5mhQX6iK87u-eplGPT9UDQD/edit?usp=drive_link&ouid=102355191132643804552&rtpof=true&sd=true)

- **Slides da Apresentação:** [Acessar no Canva](https://www.canva.com/design/DAG4dOYjrAI/Mh7PE6JILAYBYoSbdRNeDg/edit)

- **Dashboard Interativo:** [Acessar no Render](https://hotels-dashboard.onrender.com/)

---

## 🎓 Contexto Acadêmico

**Instituição:** Universidade Positivo  
**Curso:** Análise e Desenvolvimento de Sistemas  
**Disciplina:** Data Science  
**Professor:** Rubem Matimoto Koide  
**Ano:** 2025

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<div align="center">

Desenvolvido com muito ☕ e carinho

⭐ Se este projeto foi útil, deixe uma estrela!

</div>
