[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

# Sports vs Politics Text Classification System

**Author:** Priyadip Sau (M25CSA023)  
**Course:** Natural Language Understanding - Semester 2, Assignment 1  
**Date:** February 2026

---

## Overview

This project tackles a deceptively straightforward problem classification of Sports and political articles or distinguishing sports articles from political coverage or vice versa through the lens of classical machine learning. The motivation is not novelty of the task itself but rather the comparative analysis it enables. We systematically pit six classifiers against five feature extraction strategies, yielding 30 distinct experimental configurations evaluated across multiple metrics. The pipeline spans everything from raw data ingestion and template-driven synthetic augmentation to preprocessing with NLTK, vectorization via scikit-learn, and a rich visualization suite that produces publication-ready figures.

What emerges is a nuanced picture: linear models dominate this high-dimensional landscape, TF-IDF representations consistently outperform raw count vectors, and bigram features strike the optimal balance between vocabulary explosion and semantic granularity. But the devil, as always, is in the details.

## Project Structure

```
sports_politics_classifier/
├── README.md
├── requirements.txt
├── data/
│   ├── bbc-news-data.csv            # BBC dataset (manually downloaded from Kaggle)
│   ├── sports_politics_dataset.csv   # Processed dataset used for classification
│   ├── final_dataset.csv             # Combined dataset (BBC + Synthetic + NewsAPI)
│   └── dataset_analysis.txt          # Auto-generated dataset statistics report
├── src/
│   ├── classifier.py                 # Core ML pipeline: preprocessing, features, models, experiments
│   ├── data_collector.py             # Data collection: BBC loading, synthetic generation, NewsAPI
│   ├── visualization.py              # Visualization toolkit: bar charts, heatmaps, radar, confusion matrices
│   └── run_visualization.py          # Standalone script to regenerate all plots from saved results
├── results/
│   ├── experiment_results.csv        # Raw experiment metrics (30 configurations)
│   ├── summary_statistics.csv        # Aggregated per-classifier statistics
│   ├── comparison_plots.png          # Multi-panel accuracy/F1/CV/heatmap figure
│   ├── confusion_matrices.png        # Confusion matrices for best model per feature method
│   ├── accuracy_comparison.png       # Grouped bar chart of accuracy
│   ├── f1_comparison.png             # Grouped bar chart of F1 scores
│   ├── cv_scores.png                 # Cross-validation with error bars
│   ├── heatmap_accuracy.png          # Accuracy heatmap (classifiers × features)
│   ├── heatmap_f1_score.png          # F1 heatmap
│   ├── heatmap_cv_mean.png           # CV mean heatmap
│   └── radar_*.png                   # Radar charts for top classifiers
└── report/
    └── ML_Classification_Report.docx # Detailed project report
```

## Dataset

The dataset is assembled from three heterogeneous sources, each contributing a different texture to the corpus. This deliberate diversity guards against overfitting to any single writing style or topical distribution.

### Source 1: BBC News Archive

The foundational data comes from the BBC News dataset, manually downloaded from [Kaggle](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive). It contains articles spanning 2004–2005 across multiple categories; we retain only the `sport` and `politics` categories. The CSV is tab-separated with columns for category, filename, title, and content. Loading is handled by `load_bbc_csv()` in `data_collector.py`, which strips whitespace from column names, filters relevant categories, and standardizes the schema.

### Source 2: Synthetic Data Generation

To augment the corpus and stress-test classifier robustness, we generate 1,000 synthetic articles (500 per class) using a template-based approach. Separate vocabulary dictionaries for sports and politics define domain-specific nouns, verbs, adjectives, teams, institutions, and named entities. Fifteen templates per category are randomly populated with vocabulary items, producing articles of 3–8 sentences each. This is intentionally simplistic-the synthetic data serves as a controlled baseline, not a substitute for real-world complexity.

### Source 3: NewsAPI (Recent Indian News)

The third stream fetches contemporary Indian news articles via the NewsAPI.org `/v2/everything` endpoint. Sports queries target cricket, football, badminton, kabaddi, hockey, and athletics coverage, while politics queries span parliament, elections, governance, and party-level discourse. Each article undergoes HTML tag removal, truncation marker cleanup, and junk pattern stripping before inclusion. This component introduces temporal recency and regional diversity that the BBC archive lacks.

### Preprocessing Pipeline

All text undergoes a multi-stage cleaning process implemented in the `TextPreprocessor` class. The pipeline lowercases text, strips URLs and email addresses, removes standalone numbers and punctuation, and collapses whitespace. Tokenization uses NLTK's `word_tokenize` with a fallback to whitespace splitting. Stopword removal draws on the standard English stopword list. Lemmatization is applied by default via WordNet (stemming with Porter is available but disabled). The result is a cleaned, normalized representation ready for vectorization.

## Feature Representations

Five feature extraction configurations are evaluated, each instantiated through the `FeatureExtractor` class with `max_features=5000`, `min_df=2`, and `max_df=0.95`:

**Bag of Words (Unigram)** - Simple term-frequency counts over individual words. This serves as the baseline representation. No weighting is applied; frequent terms dominate the feature space regardless of their discriminative power.

**Bag of Words (Bigram)** - Extends the vocabulary to include word pairs alongside unigrams. Captures local co-occurrence patterns (e.g., "world cup", "prime minister") at the cost of a substantially larger feature space.

**TF-IDF (Unigram)** - Term Frequency–Inverse Document Frequency weighting with sublinear TF scaling (`sublinear_tf=True`). This dampens the influence of high-frequency terms and amplifies rare but informative tokens. A critical step for text classification where common words carry little class signal.

**TF-IDF (Bigram)** - Combines TF-IDF weighting with unigram-bigram features. Empirically, this configuration emerges as the strongest across most classifiers-it retains the discriminative weighting of TF-IDF while capturing meaningful two-word phrases.

**TF-IDF (Trigram)** - Pushes the n-gram range to (1,3). Trigrams introduce longer phrasal patterns but also increase sparsity substantially. The `max_features` cap at 5,000 becomes a binding constraint here, and some useful trigrams may be pruned.

## Machine Learning Models

Six classifiers are wrapped in the `TextClassifier` class, each accessible by name:

**Multinomial Naive Bayes** (`naive_bayes`) - The probabilistic workhorse of text classification. Assumes conditional independence among features-an assumption flagrantly violated in natural language, yet the model performs remarkably well regardless. No hyperparameter tuning beyond scikit-learn defaults.

**Logistic Regression** (`logistic_regression`) - L2-regularized linear classifier with `max_iter=1000`. Outputs calibrated probabilities and interpretable coefficients. In high-dimensional sparse settings like text, logistic regression is a formidable competitor to more complex models.

**Support Vector Machine** (`svm`) - Linear kernel SVM with `probability=True` enabled for ROC-AUC computation. Maximizes the margin between class boundaries in the feature space. SVMs have a long track record of strong performance on text classification tasks, particularly when the data is high-dimensional and approximately linearly separable.

**Random Forest** (`random_forest`) - An ensemble of 100 decision trees with bootstrap aggregation. Captures non-linear interactions but tends to underperform linear models on sparse text features. Included here primarily as a non-linear baseline and for its feature importance capabilities.

**Gradient Boosting** (`gradient_boosting`) - Sequential ensemble of 100 boosted trees. Iteratively corrects misclassifications from prior trees. More expressive than Random Forest but prone to overfitting on high-dimensional sparse data without careful regularization.

**K-Nearest Neighbors** (`knn`) - Instance-based classifier with `n_neighbors=5`. Makes predictions based on the majority class among the five closest training examples. Computationally expensive at inference time and typically struggles in high-dimensional spaces due to the curse of dimensionality.

## Experimental Setup

The experiment orchestration lives in the `ExperimentRunner` class. Data is split 80/20 with stratification to preserve class balance. For each of the 5 feature methods × 6 classifiers = 30 configurations, the pipeline extracts features on the training split, fits the classifier, evaluates on the held-out test set, and performs 5-fold cross-validation on the training data. Metrics collected include accuracy, precision, recall, F1-score (all weighted), ROC-AUC (where probability estimates are available), and cross-validation mean with standard deviation.

Results are aggregated into a DataFrame and persisted to `experiment_results.csv`. A comparison pivot table is also generated for quick inspection.

## Visualization Suite

The `ExperimentVisualizer` class in `visualization.py` produces a comprehensive set of figures:

Grouped bar charts compare accuracy and F1 scores across all classifier–feature combinations. A cross-validation plot overlays error bars to visualize variance. Heatmaps provide a dense summary of how each metric varies across the full grid of configurations. Confusion matrices are rendered for the best-performing classifier under each feature method, annotated with accuracy. Radar charts offer a multi-metric profile for individual classifiers across feature methods. Additionally, `plot_dataset_statistics()` generates class distribution histograms, text length distributions, word count box plots, and class proportion pie charts.

All figures are saved at 300 DPI and suitable for direct inclusion in the project report.

## Installation and Usage

### Prerequisites

Python 3.8 or higher is required. A NewsAPI key is needed only if you wish to fetch recent Indian news articles.

### Setup

```bash
git clone https://github.com//priyadip/sports-politics-classifier.git
cd sports-politics-classifier
pip install -r requirements.txt
```

Download the BBC News dataset from [Kaggle](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive) and place the CSV file at `data/bbc-news-data.csv`.

NLTK data downloads are handled automatically at runtime, but you can also trigger them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running the Data Collection Pipeline

```bash
cd src
python data_collector.py
```

This loads the BBC CSV, generates 1,000 synthetic articles, optionally fetches Indian news via NewsAPI, combines all sources into `data/final_dataset.csv`, and prints a dataset analysis report saved to `data/dataset_analysis.txt`.

### Running the Classification Experiments

```bash
cd src
python classifier.py
```

Executes all 30 experimental configurations. Results are printed to the console and saved to `results/experiment_results.csv`. Comparison plots and confusion matrices are saved to the `results/` directory.

### Regenerating Visualizations

If you already have `experiment_results.csv` and want to regenerate the full visualization suite:

```bash
cd src
python run_visualization.py
```

This produces accuracy comparisons, F1 comparisons, CV score plots, heatmaps for three metrics, confusion matrices, radar charts, and summary statistics.

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nltk>=3.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
requests
```

## License

This project is licensed under the MIT License.

---

**Note:** Developed as part of the NLU course assignment (Semester 2) exploring traditional machine learning approaches to binary text classification.
