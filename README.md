# Penguin Species Classification with KNN

This workshop uses K-Nearest Neighbors (KNN) to predict a penguin's species from its physical measurements, island, and sex. It demonstrates loading data, handling missing values, preparing features, training a classifier, and evaluating predictions.

## Project files

- `knn_penguins.ipynb` notebook containing the complete exercise.
- `README.md` project overview and setup instructions.

## Dataset

The notebook loads the Palmer Penguins dataset using `seaborn.load_dataset('penguins')`. Its saved output shows **344 rows and 7 columns**. An internet connection is needed on the first load unless the dataset is already cached.

| Column              | Purpose                                                    |
| ------------------- | ---------------------------------------------------------- |
| `bill_length_mm`    | Bill length in millimetres; numerical feature              |
| `bill_depth_mm`     | Bill depth in millimetres; numerical feature               |
| `flipper_length_mm` | Flipper length in millimetres; numerical feature           |
| `body_mass_g`       | Body mass in grams; numerical feature                      |
| `island`            | Island where the penguin was observed; categorical feature |
| `sex`               | Penguin sex; categorical feature                           |
| `species`           | Target class: Adelie, Chinstrap, or Gentoo                 |

## Setup and running

1. Place the notebook and README in the same project folder.
2. Open that folder in VS Code and create a virtual environment from the terminal:

   ```powershell
   py -m venv .venv
   ```

3. Install the packages into that environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install pandas numpy seaborn "scikit-learn>=1.2" jupyter ipykernel
   ```

   The notebook uses `OneHotEncoder(sparse_output=False)`, which requires scikit-learn 1.2 or later.

4. Install the Python and Jupyter extensions in VS Code if needed.
5. Open `knn_penguins.ipynb`, select the `.venv` Python interpreter as the notebook kernel, and choose **Run All**.

Alternatively, launch Jupyter from the projeinstact folder:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

## How the exercise works

1. **Load data:** `load_data()` retrieves the penguin dataset.
2. **Prepare features:** `preprocess_data()` removes rows with a missing species, separates inputs (`X`) from the target (`y`), and applies different transformations to numerical and categorical columns.
3. **Handle missing values:** Numerical values are filled with each column's median. Categorical values are filled with each column's most frequent category.
4. **Scale and encode:** `StandardScaler` standardizes numerical features so differences in measurement units do not dominate distances. One-hot encoding converts categories into separate numeric columns.
5. **Split data:** 80% is used for training and 20% for testing. `random_state=42` makes the split reproducible; `stratify=y` approximately preserves each species' proportion in both sets.
6. **Train:** `train_model()` creates a classifier with `k=5`. KNN predicts a species by finding the five nearest training examples and taking their majority vote.
7. **Evaluate:** `evaluate_model()` predicts the test labels and prints accuracy and a classification report.

## Results in the saved notebook

The existing notebook output reports **0.9855 accuracy (98.55%)** across **69 test examples**, equivalent to 68 correct predictions. This is the saved result, not a newly rerun experiment.

The classification report includes:

- **Precision:** Of the penguins predicted as a species, how many actually belong to that species?
- **Recall:** Of the penguins that actually belong to a species, how many were identified correctly?
- **F1-score:** A combined measure of precision and recall.
- **Support:** The number of test examples for each species.

## Evaluation limitation

The current notebook fits its preprocessing transformations on the entire dataset **before** splitting it into training and test sets. This allows test-set information to influence the imputation and scaling statistics, creating data leakage. The saved accuracy should therefore be treated as a workshop result rather than a fully independent estimate of model performance.

For a more reliable evaluation, split the original features first, fit the preprocessor only on the training set, and transform the test set with that fitted preprocessor. A single pipeline combining preprocessing and KNN can enforce this order when fitted on training data.

## Practice ideas

- Compare `k=3`, `k=5`, and `k=7` using a validation set or cross-validation within the training data.
- Explain why scaling matters when body mass is measured in grams and bill length in millimetres.
- Inspect which species are confused and discuss possible reasons.
- Correct the preprocessing order and compare the resulting evaluation with the saved output.
