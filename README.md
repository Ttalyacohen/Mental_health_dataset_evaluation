# Social Media Usage and Student Mental Health – Statistical Analysis and ML

A data science project analyzing how social media usage relates to students' stress, mental health and quality of life, using statistical hypothesis testing and machine learning models.

Team project with [Neriya Filber](https://github.com/NeriyaFilber) and [Talya Cohen](https://github.com/Ttalyacohen).

## Dataset

*Student Social Media and Mental Health Impact*: about 5,000 students with demographics, platform and purpose of use, daily usage hours, daily phone unlocks, study, sleep and physical activity hours, stress level (4 levels) and a mental health score.

## Research questions and key results

| # | Question | Methods | Main result |
|---|---|---|---|
| 1 | Does daily usage correlate with mental health? | Pearson/Spearman, linear regression | Strong negative correlation (r ≈ −0.8 for both genders); usage alone explains 67% of the variance |
| 2 | Do men and women differ in stress? | Normality and group tests, effect sizes, logistic regression | Usage predicts high stress (accuracy 0.888, ROC-AUC 0.962); adding gender adds almost nothing |
| 3 | Does stress differ by purpose of use? | Chi-square test, Cramér's V | Statistically significant but practically small association (V = 0.066) |
| 4 | Can the purpose of use be predicted? | Logistic Regression, SVM, Random Forest, XGBoost, tuning | XGBoost macro F1 0.981, but a platform-only rule already reaches 0.945: the platform drives the prediction |
| 5 | Can daily usage hours be predicted? | Ridge, KNN, SVR, Random Forest, permutation importance | Random Forest R² 0.943 |
| 6 | Is usage related to quality of life? | Pearson/Spearman, Kruskal-Wallis | Usage is strongly negatively correlated with study hours (r = −0.879) |
| 7 | Do quality-of-life features relate to stress? | Logistic and linear regression, ROC | High stress predicted with CV accuracy 0.884 and ROC-AUC 0.959 |
| 8 | Can stress level and mental health score be predicted? | 8 classifiers and 6 regressors with grid search, soft and hard voting ensembles | Stress (4 levels): accuracy 0.914 vs a 0.324 baseline; mental health score: R² 0.889 (KNN) |

Data exploration also includes distributions, correlation analysis, PCA, t-SNE and model-based feature selection.

**Main takeaways:** usage behavior strongly predicts both stress and mental health, non-linear models consistently beat linear ones, and some high scores come from shortcut features (such as platform → purpose), which the analysis checks against simple baselines.

## Project structure

```
dataset/            raw and cleaned data
data_exploration/   data overview, distributions, feature relationships, PCA / t-SNE
Questions/          one notebook per research question (01–08)
pyproject.toml      dependencies
```

## Run

```bash
pip install pandas matplotlib seaborn scikit-learn scipy xgboost plotly geopandas pycountry geodatasets jupyterlab
jupyter lab
```

Open any notebook under `Questions/` or `data_exploration/`.
