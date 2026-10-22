# Amazon Baby Review Sentiment Classification

An NLP and machine learning project that classifies Amazon baby-product reviews as positive or negative using Logistic Regression and a bag-of-words representation.

The project compares a model using the full vocabulary learned from training data with a model restricted to a predefined list of 20 words. The analysis focuses on predictive performance, the effect of vocabulary size, and the limitations of sentiment classification based on star ratings.

This project started as an academic coursework exercise and was refactored and extended with additional evaluation metrics, a majority-class baseline, and a more structured analysis.

## Research Questions

- How well can a Logistic Regression model classify positive and negative reviews?
- What is the effect of restricting the vocabulary to 20 predefined words?
- How does the smaller vocabulary affect training time and the detection of negative reviews?
- What are the limitations of a bag-of-words approach?

## Dataset

The dataset contains 183,531 product reviews with three columns:

- `name` - product name
- `review` - customer review text
- `rating` - star rating from 1 to 5

For this experiment, the target is defined as follows:

- Ratings 1–2 → negative sentiment
- Ratings 4–5 → positive sentiment
- Rating 3 → excluded as neutral

After excluding neutral ratings and empty reviews, 165,975 reviews remained: 139,582 positive and 26,393 negative.

## Source and Access
The dataset is associated with the University of Washington's *Machine Learning Foundations: A Case Study Approach* course materials, which include an assignment on product sentiment analysis.

A publicly available repository documenting these course materials references **amazon_baby.csv** for the Week 3 sentiment-analysis assignment:

- *[Machine Learning Foundations — University of Washington course materials](https://github.com/raj-preacher/Machine-Learning-WASHINGTON-Course-1)*

The dataset file is not included in this repository. To reproduce the experiment, the CSV must be placed in the same directory as the notebook and named amazon_baby.csv.

## Methodology

1. Inspect the dataset and prepare the binary sentiment target.
2. Split the data into training and test sets using stratified sampling (80/20).
3. Establish a majority-class baseline.
4. Train Logistic Regression using a full CountVectorizer vocabulary.
5. Train Logistic Regression using a predefined vocabulary of 20 words.
6. Compare accuracy, precision, recall, F1-score, macro F1-score, and training time.
7. Inspect influential word coefficients and the model's classification errors.

The vectorizer is fitted on the training data only. The test data is transformed using the vocabulary learned from the training set.

## Results

Results from the current experiment using `random_state=42`:

| Metric | Majority baseline | Full vocabulary | 20 words |
|---|---:|---:|---:|
| Accuracy | 84.10% | **93.26%** | 86.95% |
| F1 - positive | 0.9136 | **0.9603** | 0.9269 |
| Recall - negative | 0.0000 | **0.7348** | 0.2650 |
| F1 - negative | 0.0000 | **0.7762** | 0.3924 |
| Macro F1 | 0.4568 | **0.8683** | 0.6596 |
| Training time | - | 16.28 s | 0.23 s |

Training times are single-run measurements from the current environment and may vary across machines.

### Key Findings

- The full-vocabulary model achieved 93.26% accuracy, compared with 84.10% for the majority-class baseline.
- Restricting the vocabulary to 20 words reduced training time substantially, but accuracy decreased to 86.95%.
- The compact model detected only 26.5% of negative reviews, compared with 73.48% for the full-vocabulary model.
- Macro F1 highlights the performance difference across both sentiment classes, which accuracy alone can hide in an imbalanced dataset.

The results demonstrate a trade-off between computational efficiency and classification quality. For this dataset, the full-vocabulary model is the stronger baseline, particularly for identifying negative reviews.

## Limitations

- **Class imbalance:** Positive reviews represent approximately 84% of the retained dataset.
- **Rating-derived labels:** Star ratings are proxies for sentiment and may be subjective or inconsistent.
- **Bag-of-words representation:** The model does not reliably capture word order, context, sarcasm, or negation.
- **Predefined vocabulary:** The 20-word list is specified in advance and may not generalize to other review datasets.
- **Dataset-specific performance:** The results should not be interpreted as a guarantee of performance on other products or customer populations.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── amazon-baby-review-sentiment.ipynb
```

## How to Run

Python 3.11 or newer is recommended. The current notebook was run in a Python 3.14 environment.

1. Clone the repository and move into the project directory.
2. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Place `amazon_baby.csv` in the same directory as the notebook. The dataset is not distributed with this repository.
5. Launch the notebook:

```bash
jupyter notebook amazon-baby-review-sentiment.ipynb
```

6. Select **Kernel → Restart Kernel and Run All Cells** to reproduce the analysis.

## Future Work

Potential extensions include TF-IDF features, word n-grams, class weighting, and more systematic error analysis.

The goal of this project is to demonstrate a transparent and reproducible machine learning experiment, rather than a deployed sentiment-analysis service.
