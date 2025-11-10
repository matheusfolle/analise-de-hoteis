# Impact of Digital Presence on Brazilian Hotel Performance

This data science project analyzes a dataset of 10,316 Brazilian hotels to understand the complex relationship between a hotel's digital presence (`has_website`), its customer engagement/popularity (`reviewsCount`), and its perceived performance/quality (`totalScore`).

The project follows a rigorous 3-phase data science pipeline, moving from statistical analysis to predictive modeling.

## 🎯 Key Questions

1.  Does having a website correlate with higher review scores and more engagement?
2.  Is a hotel's popularity (review count) a good predictor of its quality (score)?
3.  Conversely, can a hotel's performance profile (its combined score and review count) predict its digital strategy (whether it has a website)?

---

## 🔬 The 3-Phase Analysis Pipeline

### Phase 1: Descriptive Statistics

Initial analysis confirmed that digital presence is strongly associated with superior performance. Hotels _with_ a website, on average, showed:

- **+767%** more reviews (`reviewsCount`)
- **17%** higher scores (`totalScore`)

### Phase 2: Regression Analysis (A Finding in "Failure")

This phase rigorously tested if popularity (`reviewsCount`) could predict quality (`totalScore`).

- **Key Finding:** The variables are **statistically independent**.
- Six different regression models (including Linear, Non-Linear Exponential, and Bayesian MCMC) all failed to find a meaningful relationship, yielding a **Coefficient of Determination (R²) of ≈0.0003**.
- **Conclusion:** This _proves_ that a hotel's popularity is not a reliable predictor of its quality in this dataset.

### Phase 3: Machine Learning (The "Methodological Pivot")

Given the independence found in Phase 2, the problem was reformulated. Instead of trying `X -> Y`, we tested if `(X + Y) -> Z`.

We used `totalScore` and `reviewsCount` as combined features to predict the `has_website` target variable.

- **Success:** This approach was highly successful. The classification models proved that a hotel's performance profile is a strong predictor of its business strategy.

---

## 🏆 Winning Model & Conclusion

Four classification algorithms (Decision Tree, Random Forest, KNN, Neural Network) were trained and evaluated.

- **Winning Model:** **Decision Tree**
- **Metric:** **76.3% Accuracy**

**Final Conclusion:** While popularity and quality are independent, a hotel's operational performance (its score) and its market engagement (its review volume) are **deeply intertwined** with its digital presence strategy.

## 📂 Project Structure
analise-de-hoteis/

│

├── data/

│   └── (.xlsx files)

│

├── docs/

│   └── Relatório.docx
|   └── Infográfico.pdf

│

├── notebooks/

│   ├── 01_descriptive_analysis.ipynb

│   ├── 02_regression_analysis.ipynb

│   └── 03_machine_learning.ipynb

│

├── src/

│   ├── init.py

│   └── analysis.py (Data validation scripts)

│

└── README.md

## 💻 Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn (for Regression, Clustering, and Classification models)
* Matplotlib & Seaborn (for visualization)
* Jupyter Notebooks