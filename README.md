# Palmer Penguins: Predicting Species from Body Measurements

Ironhack Data Analytics bootcamp, Unit 7 machine learning project (individual).

## Question
Can a penguin's species be predicted from four body measurements alone? And, as an additional test, can its sex?

## Dataset
- **Source:** [Palmer Archipelago (Antarctica) penguin data](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) on Kaggle (`penguins_size.csv`). The data was originally collected by Dr. Kristen Gorman at Palmer Station, Antarctica LTER.
- **Size:** 344 penguins, 333 after cleaning.
- **Species:** Adelie, Chinstrap, Gentoo.
- The dataset is not stored in this repo. The notebook downloads it with `kagglehub`.

## Repository structure
| File | Content |
|---|---|
| `[notebook_name].ipynb` | Full workflow: cleaning, exploration, models, tuning, evaluation |
| `functions.py` | Reusable functions (e.g. `plot_confusion_map()`) |
| `README.md` | This file |

Presentation slides: [link]

## Workflow
1. **Cleaning:** removed rows with missing values and one invalid `"."` entry in `sex`.
2. **Exploration:** pairplot and correlation heatmap.
3. **Feature selection:** culmen (bill) length, culmen depth, flipper length, body mass. `island` was left out on purpose: some islands host only one species, so the model could guess from location instead of the penguin's body.
4. **Preparation:** 80/20 train-test split (`random_state=0`), MinMaxScaler fitted on the training set only.
5. **Models:** KNN and Decision Tree, plus four ensemble methods: Bagging, Random Forest, AdaBoost, Gradient Boosting.
6. **Tuning:** Grid Search with 5-fold cross-validation on KNN (`n_neighbors`, `weights`).
7. **Evaluation:** accuracy, classification report, confusion matrices, and inspection of the misclassified penguins.

## Results: species
| Model | Test accuracy |
|---|---|
| KNN (k=5), Decision Tree, Bagging, Random Forest, Gradient Boosting | 97.0% (2 wrong out of 67) |
| AdaBoost | 95.5% (3 wrong) |
| Tuned KNN (k=3) | 98.5% (1 wrong) |
| Baseline: always predict Adelie | 58.2% on the test set (43.8% on the full dataset) |

- All models land in the same place, so the measurements do the work, not the algorithm.
- Bill length is the most important feature (Random Forest feature importance).
- **The errors:** both are Chinstraps predicted as Adelie, with short bills (40.9 and 43.5 mm), where the two species overlap. Chinstrap is the smallest class and has the lowest recall (0.80).
- Tuning removed one of the two errors. On a 67-penguin test set this is a single bird, so it is a small gain, not proof of a better model.

## Additional test: sex
- KNN (k=5): 95.5% test accuracy (3 wrong out of 67), against a 61.2% baseline (always predict female on this test set).
- Tuned KNN: 89.9% in 5-fold cross-validation. This is the more reliable number, because one penguin moves the test score by 1.5 points.
- The misclassified penguins are large Adelie females predicted as male.

## Limitations
- **Small data:** 333 penguins, 67 in the test set, so single-test-set scores are noisy.
- **Unrepresentative test set:** the split was not stratified, so the test set is 58% Adelie and 61% female, while the full dataset is 44% Adelie and roughly balanced by sex.
- **Narrow scope:** three species from one small region; the model would not generalise to other penguin populations.
- **No derived features** were engineered beyond scaling.

## Possible applications
Supporting field researchers in ecological monitoring, e.g. classifying birds from measurements when visual identification is uncertain.

## How to run
1. Clone the repo: `git clone [repo link]`
2. Install the libraries: `pip install pandas scikit-learn matplotlib seaborn kagglehub`
3. Open the notebook and run all cells. The first cells download the dataset from Kaggle.
