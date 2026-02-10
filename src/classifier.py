"""
Sports vs Politics Text Classifier

A comprehensive text classification system comparing multiple ML techniques
with various feature representation methods.
By:
Priyadip Sau (M25CSA023)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """
    Handles all text preprocessing operations including cleaning,
    tokenization, stemming, and lemmatization.
    """
    
    def __init__(self, use_stemming=False, use_lemmatization=True, 
                 remove_stopwords=True):
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Remove noise from text data."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (but keep words with numbers like '2024')
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess(self, text):
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Stemming or Lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)


class FeatureExtractor:
    """
    Implements multiple feature extraction techniques:
    - Bag of Words (BoW)
    - TF-IDF
    - N-grams (unigrams, bigrams, trigrams)
    """
    
    def __init__(self, method='tfidf', ngram_range=(1, 1), max_features=5000):
        self.method = method
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = None
        
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts to feature matrix."""
        if self.method == 'bow':
            self.vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95
            )
        elif self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Return feature names from vectorizer."""
        return self.vectorizer.get_feature_names_out()


class TextClassifier:
    """
    Main classifier class that wraps multiple ML algorithms
    for text classification tasks.
    """
    
    CLASSIFIERS = {
        'naive_bayes': MultinomialNB(),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'svm': SVC(kernel='linear', probability=True, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5)
    }
    
    def __init__(self, classifier_name='naive_bayes'):
        if classifier_name not in self.CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        self.classifier_name = classifier_name
        self.model = self.CLASSIFIERS[classifier_name]
        
    def train(self, X_train, y_train):
        """Train the classifier."""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_test)
        return None
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation of the model."""
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC-AUC if probabilities available
        y_proba = self.predict_proba(X_test)
        if y_proba is not None:
            results['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
        return results


class ExperimentRunner:
    """
    Orchestrates the entire experimental pipeline:
    - Data loading and preprocessing
    - Feature extraction with different methods
    - Training and evaluation of multiple classifiers
    - Results aggregation and visualization
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.results = {}
        
    def load_data(self, data_path=None, data=None):
        """Load data from file or DataFrame."""
        if data is not None:
            self.df = data
        elif data_path is not None:
            if data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                self.df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            raise ValueError("Either data_path or data must be provided")
        
        return self
    
    def preprocess_data(self, text_column='text', label_column='label'):
        """Preprocess all text data."""
        print("Preprocessing text data...")
        self.df['processed_text'] = self.df[text_column].apply(
            self.preprocessor.preprocess
        )
        self.texts = self.df['processed_text'].values
        self.labels = self.df[label_column].values
        
        # Encode labels if they're strings
        if self.labels.dtype == 'object':
            self.label_mapping = {label: idx for idx, label in 
                                 enumerate(np.unique(self.labels))}
            self.labels = np.array([self.label_mapping[l] for l in self.labels])
        
        return self
    
    def run_experiment(self, feature_methods, classifiers, test_size=0.2):
        """Run full experimental pipeline."""
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            self.texts, self.labels, test_size=test_size, 
            random_state=self.random_state, stratify=self.labels
        )
        
        all_results = []
        
        for feat_config in feature_methods:
            print(f"\n{'='*60}")
            print(f"Feature Method: {feat_config['name']}")
            print(f"{'='*60}")
            
            # Extract features
            extractor = FeatureExtractor(
                method=feat_config['method'],
                ngram_range=feat_config.get('ngram_range', (1, 1)),
                max_features=feat_config.get('max_features', 5000)
            )
            
            X_train = extractor.fit_transform(X_train_text)
            X_test = extractor.transform(X_test_text)
            
            for clf_name in classifiers:
                print(f"\n  Training {clf_name}...")
                
                # Train classifier
                classifier = TextClassifier(clf_name)
                classifier.train(X_train, y_train)
                
                # Evaluate
                eval_results = classifier.evaluate(X_test, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    classifier.model, X_train, y_train, cv=5, scoring='accuracy'
                )
                
                result = {
                    'feature_method': feat_config['name'],
                    'classifier': clf_name,
                    'accuracy': eval_results['accuracy'],
                    'precision': eval_results['precision'],
                    'recall': eval_results['recall'],
                    'f1_score': eval_results['f1_score'],
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'confusion_matrix': eval_results['confusion_matrix']
                }
                
                if 'roc_auc' in eval_results:
                    result['roc_auc'] = eval_results['roc_auc']
                
                all_results.append(result)
                
                print(f"    Accuracy: {eval_results['accuracy']:.4f}")
                print(f"    F1 Score: {eval_results['f1_score']:.4f}")
                print(f"    CV Mean:  {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def generate_comparison_table(self):
        """Generate formatted comparison table."""
        if self.results is None or len(self.results) == 0:
            return None
        
        pivot = self.results.pivot_table(
            index='classifier',
            columns='feature_method',
            values=['accuracy', 'f1_score', 'cv_mean']
        )
        
        return pivot
    
    def plot_results(self, save_path=None):
        """Generate visualization plots."""
        if self.results is None or len(self.results) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        pivot_acc = self.results.pivot(
            index='classifier', columns='feature_method', values='accuracy'
        )
        pivot_acc.plot(kind='bar', ax=ax1)
        ax1.set_title('Accuracy Comparison Across Methods', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Classifier')
        ax1.legend(title='Feature Method', bbox_to_anchor=(1.02, 1))
        ax1.set_ylim([0.85, 0.92])
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: F1 Score comparison
        ax2 = axes[0, 1]
        pivot_f1 = self.results.pivot(
            index='classifier', columns='feature_method', values='f1_score'
        )
        pivot_f1.plot(kind='bar', ax=ax2)
        ax2.set_title('F1 Score Comparison Across Methods', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_xlabel('Classifier')
        ax2.legend(title='Feature Method', bbox_to_anchor=(1.02, 1))
        ax2.set_ylim([0.85, 0.92])
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: CV Mean with error bars
        ax3 = axes[1, 0]
        for feat in self.results['feature_method'].unique():
            subset = self.results[self.results['feature_method'] == feat]
            x = range(len(subset))
            ax3.errorbar(x, subset['cv_mean'], yerr=subset['cv_std'], 
                        label=feat, marker='o', capsize=3)
        ax3.set_xticks(range(len(self.results['classifier'].unique())))
        ax3.set_xticklabels(self.results['classifier'].unique(), rotation=45, ha='right')
        ax3.set_title('Cross-Validation Scores with Standard Deviation', fontsize=12, fontweight='bold')
        ax3.set_ylabel('CV Score')
        ax3.set_xlabel('Classifier')
        ax3.legend(title='Feature Method')
        ax3.set_ylim([0.85, 0.92])
        
        # Plot 4: Heatmap of all metrics
        ax4 = axes[1, 1]
        heatmap_data = self.results.pivot(
            index='classifier', columns='feature_method', values='accuracy'
        )
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax4)
        ax4.set_title('Accuracy Heatmap', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, save_path=None):
        """Plot confusion matrices for best models."""
        if self.results is None or len(self.results) == 0:
            return
        
        # Get unique feature methods
        feature_methods = self.results['feature_method'].unique()
        n_methods = len(feature_methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        if n_methods == 1:
            axes = [axes]
        
        for idx, feat in enumerate(feature_methods):
            # Get best classifier for this feature method
            subset = self.results[self.results['feature_method'] == feat]
            best_idx = subset['accuracy'].idxmax()
            best_row = self.results.loc[best_idx]
            
            cm = best_row['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f"{feat}\n({best_row['classifier']})", fontsize=10)
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.suptitle('Confusion Matrices for Best Models per Feature Method', 
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    """Main execution function."""
    
    # Define feature extraction methods to compare
    feature_methods = [
        {'name': 'BoW (Unigram)', 'method': 'bow', 'ngram_range': (1, 1)},
        {'name': 'BoW (Bigram)', 'method': 'bow', 'ngram_range': (1, 2)},
        {'name': 'TF-IDF (Unigram)', 'method': 'tfidf', 'ngram_range': (1, 1)},
        {'name': 'TF-IDF (Bigram)', 'method': 'tfidf', 'ngram_range': (1, 2)},
        {'name': 'TF-IDF (Trigram)', 'method': 'tfidf', 'ngram_range': (1, 3)},
    ]
    
    # Define classifiers to compare
    classifiers = [
        'naive_bayes',
        'logistic_regression', 
        'svm',
        'random_forest',
        'gradient_boosting'
    ] 
    print("SPORTS VS POLITICS TEXT CLASSIFIER")
    # Initialize experiment runner
    runner = ExperimentRunner(random_state=42)    
    os.makedirs('../results', exist_ok=True)

    df = pd.read_csv('../data/sports_politics_dataset.csv')
    df.columns = df.columns.str.strip()

    df['text'] = df['title'].fillna('') + " " + df['content'].fillna('')
    df['label'] = df['category']

    runner.load_data(data=df)

    # Preprocess
    runner.preprocess_data(text_column='text', label_column='label')
    
    # Run experiments
    results = runner.run_experiment(feature_methods, classifiers, test_size=0.2)
    
    # Display results

    print("FINAL RESULTS ")
    comparison = runner.generate_comparison_table()
    print("\nComparison Table:")
    print(comparison)
    
    # Find best combination
    best_idx = results['accuracy'].idxmax()
    best = results.loc[best_idx]
    print(f"\nBest Performing Combination:")
    print(f"  Feature Method: {best['feature_method']}")
    print(f"  Classifier: {best['classifier']}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  F1 Score: {best['f1_score']:.4f}")
    
    # Generate plots
    runner.plot_results(save_path='../results/comparison_plots.png')
    runner.plot_confusion_matrices(save_path='../results/confusion_matrices.png')
    
    # Save results
    results.to_csv('../results/experiment_results.csv', index=False)
    print("\nResults saved to ../results/experiment_results.csv")
    
    return results


if __name__ == "__main__":
    results = main()
