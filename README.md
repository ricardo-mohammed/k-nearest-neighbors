# K-Nearest Neighbors (KNN) Workshop — Palmer Penguins Classification

## Project Overview

This workshop uses **K-Nearest Neighbors (KNN)** to classify penguins as **Adelie, Chinstrap, or Gentoo** based on their measurements and other recorded characteristics. KNN looks at the most similar examples in the training set and predicts the most common class among their nearest neighbors.

Our main goal was not only to obtain a high accuracy, but also to understand how data preparation affects a distance-based model. We explored the dataset, handled missing values, converted categorical information into numbers, scaled numerical features, trained KNN, and evaluated the results. We organized these steps with a reusable machine learning pipeline.

**Main experiment:** Palmer Penguins (344 observed penguins; this is a real-world observational dataset, not synthetic data). The Iris notebook provided by the instructor is retained separately as a learning exercise.

## Implementation

The final implementation and saved model results are in **`notebooks/knn_penguins.ipynb`**. The notebook imports `inspect_penguins` from `src/EDA.py` for its exploratory data analysis (EDA).

## Dataset and Classification Problem

**Dataset:** Palmer Penguins  
**Source:** Palmer Archipelago (Antarctica) research dataset, distributed through the `palmerpenguins` project and available through Seaborn.  
**Dataset reference:** https://github.com/allisonhorst/palmerpenguins  

## Environment Setup and Execution

**Tools:** Python, VS Code / Jupyter Notebook, pandas, NumPy, Seaborn, matplotlib, and scikit-learn.

From the project root in **PowerShell**:

```powershell
# Create a local Python environment if one does not exist.
py -m venv .venv

# Activate it.
.\.venv\Scripts\Activate.ps1

# Install the dependencies already listed in this repository.
python -m pip install -r requirements.txt
```

Open `notebooks/knn_penguins.ipynb` in VS Code, select the `.venv` Python kernel, and execute the cells in order. Run VS Code **from the project root** so that the notebook can import `src.EDA` correctly. The notebook's data loader uses `sns.load_dataset('penguins')`, so the first download might need an internet connection.

**Execution note for the current attached notebook:** Its EDA cell calls `inspect_penguins(penguins_df)` **before** the later cell that creates `penguins_df`. If starting with a clean kernel, load the dataframe before running the EDA cell, for example:

```python
penguins_df = sns.load_dataset('penguins')
inspect_penguins(penguins_df)
```

These two lines belong **after the imports and before the EDA call**. Without this correction, running the attached notebook from top to bottom in a fresh kernel can cause a `NameError`. This is a notebook-order issue, separate from the KNN model itself.

## ML Pipeline: What We Implemented

The workflow in `knn_penguins.ipynb` is:

1. **Load the data:** `load_data()` retrieves the Penguins observations through Seaborn.
2. **Inspect the dataset:** `inspect_penguins()` summarizes its shape, missing values, numeric features, class distribution, and duplicates.
3. **Separate inputs and target:** `X` contains the six input features; `y` contains `species`.
4. **Split first:** `train_test_split(test_size=0.2, random_state=42, stratify=y)` creates training and test sets while approximately maintaining the species proportions.
5. **Build the preprocessing pipeline:** `build_preprocessor()` creates a `ColumnTransformer` with different steps for numeric and categorical features.
6. **Train KNN:** a `Pipeline` combines the preprocessor with `KNeighborsClassifier(n_neighbors=5)`; `pipeline.fit(X_train, y_train)` fits all learned transformations **only on training data**.
7. **Evaluate:** `evaluate_model()` prints test accuracy and the classification report. A separate cell creates a confusion matrix.

## Peer Review and Comparison — KNN Workshop

**Second group's project:** [Namypark / KNearestNeighbors_Workshop](https://github.com/Namypark/KNearestNeighbors_Workshop). We reviewed their provided `KNN_Workshop_Solution.ipynb` and `README.md`. Their notebook classifies **Cammeo vs. Osmancik rice grains** using the UCI Rice dataset (`fetch_ucirepo(id=545)`). This is a comparison of **documented experiments**, not a claim that we interviewed their group.

Both groups worked on a K-Nearest Neighbors classification workshop. Our project predicts the three Palmer Penguins species using numerical measurements and the categorical variables `island` and `sex`. Our code uses a train/test split, imputes missing values, scales numerical features, one-hot encodes categorical features, and trains KNN with `k=5`.

| Category | Our group: Palmer Penguins | Peer group: Rice (Cammeo / Osmancik) |
| --- | --- | --- |
| Source and records | Palmer Penguins through `seaborn.load_dataset('penguins')`; **344 rows** | UCI ML Repository via `fetch_ucirepo(id=545)`; **3,810 rows** |
| Classification target | `species`: **Adelie, Chinstrap, Gentoo** (3 classes) | `Class`: **Cammeo, Osmancik** (2 classes) |
| Input features | 4 numerical body measurements; 2 categorical fields (`island`, `sex`) | 7 numerical grain-shape measurements, including Area, Perimeter and Eccentricity |
| Missing values | Imputed using median for numeric and most frequent for categorical values | No missing values or duplicate rows reported in their notebook |
| Encoding | One-hot encoding for `island` and `sex` | Label encoding of the class target; no categorical input features |
| Scaling and data leakage | `StandardScaler` inside a `Pipeline` fitted **after** train/test split | `StandardScaler` fitted on training data, applied to test; scaler placed inside `Pipeline` for cross-validation |
| Model and split | KNN, `k=5`, stratified 80/20 split | KNN, `k=5`, stratified 80/20 split |
| Accuracy (saved test output) | **98.55%**, 68/69 correct | **90.94%**, 693/762 correct |
| Per-class results | Adelie recall 0.97; Chinstrap recall 1.00; Gentoo recall 1.00 | Cammeo recall 0.88; Osmancik recall 0.93 |
| Additional experiments | Confusion matrix and classification report | Scaling comparison, multiple values of `k`, grid search and 5-fold cross-validation |


**What we noticed:** Both projects apply the same algorithm, but to different classification problems. Our accuracy is higher; however, it is **not a fair head-to-head performance ranking** because the number of classes, sample sizes and feature distributions differ. Our test set has only 69 penguins, whereas theirs has 762 rice grains. A change of one error has a much larger effect on our percentage.

**Preprocessing comparison:** We handled missing measurements and categorical input features, which made our preprocessing more varied. Their seven inputs were already numerical and had no missing values, so their key transformation was scaling. They tested it: their KNN accuracy improved from **86.48% without scaling to 90.94% with scaling**, about **4.46 percentage points**. This gives experimental support to our explanation that distance-based models should not let large-scale features dominate smaller ones.

**Evaluation comparison:** Their confusion matrix showed **39 Cammeo grains classified as Osmancik** and **30 Osmancik grains classified as Cammeo**. Their discussion connects the mistakes to similar grain shapes and an uneven class balance. In our test set, **one Adelie penguin was classified as Chinstrap**, while all Gentoo test examples were classified correctly. In both projects, the confusion matrix is more informative than accuracy alone.

**Strengths, limits, and MLOps:** Both studies show that KNN is approachable for small or medium datasets with meaningful distances, but may be expensive at prediction time, sensitive to scaling and affected by high-dimensional or imbalanced data. We can improve reproducibility by adding tests that verify preprocessing uses only training data, saving the entire fitted pipeline, recording metrics across runs, and monitoring new data for drift. These are **future ideas**, not implemented MLOps features.


## References

- Horst, A. M., Hill, A. P., & Gorman, K. B. *palmerpenguins* dataset and documentation: https://github.com/allisonhorst/palmerpenguins
- KNN and scikit-learn pipeline documentation: https://scikit-learn.org/stable/modules/neighbors.html and https://scikit-learn.org/stable/modules/compose.html#pipeline
- Workshop starter instructions: https://github.com/ProfEspinosaAIML/KNearestNeighbors_Workshop
- Peer KNN workshop repository: https://github.com/Namypark/KNearestNeighbors_Workshop


<br/>
This project was developed with the assistance of **Qwen3.7**, a large language model developed by Alibaba Cloud. The AI provided support with code scaffolding, ML pipeline architecture design, debugging, and generating explanatory Markdown documentation for the K-Nearest Neighbors (KNN) workshop. All final code, logic, and reflections were reviewed and validated by the student authors.