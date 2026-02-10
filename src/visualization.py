"""
Visualization and Analysis Module
==================================
Comprehensive visualization toolkit for ML experiment analysis,
including performance comparisons, confusion matrices, learning curves,
and feature importance analysis.

Author: [Your Name]
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import ast

from sklearn.model_selection import learning_curve as sk_learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ExperimentVisualizer:
    """
    Comprehensive visualization toolkit for ML experiments.
    Generates publication-quality figures for reports.
    """
    
    def __init__(self, results_df: pd.DataFrame = None, figsize: tuple = (12, 8)):
        self.results = results_df
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
        
    def plot_accuracy_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Generate grouped bar chart comparing accuracy across methods.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot = self.results.pivot(
            index='classifier', 
            columns='feature_method', 
            values='accuracy'
        )
        
        x = np.arange(len(pivot.index))
        width = 0.15
        multiplier = 0
        
        for col, color in zip(pivot.columns, self.colors):
            offset = width * multiplier
            bars = ax.bar(x + offset, pivot[col], width, label=col, color=color)
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8, rotation=90)
            multiplier += 1
        
        ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Classification Accuracy: Comparison of Feature Methods and Classifiers',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.legend(title='Feature Method', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_ylim([0.85, 0.92])
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_f1_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Generate F1 score comparison visualization.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot = self.results.pivot(
            index='classifier',
            columns='feature_method',
            values='f1_score'
        )
        
        pivot.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('F1 Score Comparison Across Experimental Configurations',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Feature Method', bbox_to_anchor=(1.02, 1))
        ax.set_ylim([0.85, 0.92])
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_cv_scores_with_variance(self, save_path: str = None) -> plt.Figure:
        """
        Plot cross-validation scores with error bars showing variance.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classifiers = self.results['classifier'].unique()
        feature_methods = self.results['feature_method'].unique()
        
        x = np.arange(len(classifiers))
        width = 0.8 / len(feature_methods)
        
        for i, method in enumerate(feature_methods):
            subset = self.results[self.results['feature_method'] == method]
            subset = subset.set_index('classifier').reindex(classifiers)
            
            ax.bar(
                x + i * width - (len(feature_methods) - 1) * width / 2,
                subset['cv_mean'],
                width,
                yerr=subset['cv_std'],
                label=method,
                color=self.colors[i],
                capsize=3,
                error_kw={'elinewidth': 1.5, 'capthick': 1.5}
            )
        
        ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cross-Validation Score', fontsize=12, fontweight='bold')
        ax.set_title('5-Fold Cross-Validation Performance with Standard Deviation',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.legend(title='Feature Method', bbox_to_anchor=(1.02, 1))
        ax.set_ylim([0.85, 0.92])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_heatmap(self, metric: str = 'accuracy', save_path: str = None) -> plt.Figure:
        """
        Generate heatmap visualization of metrics.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pivot = self.results.pivot(
            index='classifier',
            columns='feature_method',
            values=metric
        )
        vmin = pivot.min().min()
        vmax = pivot.max().max()

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap='RdBu_r',
            center=(vmin + vmax) / 2,
            linewidths=0.5,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap: Feature Methods × Classifiers',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Feature Method', fontsize=11)
        ax.set_ylabel('Classifier', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list,
                             title: str = None, save_path: str = None) -> plt.Figure:
        """
        Plot a single confusion matrix.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_multiple_confusion_matrices(self, save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrices for best models per feature method.
        """
        feature_methods = self.results['feature_method'].unique()
        n_methods = len(feature_methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))
        if n_methods == 1:
            axes = [axes]
        
        class_names = ['Politics', 'Sports']
        
        for idx, method in enumerate(feature_methods):
            subset = self.results[self.results['feature_method'] == method]
            best_row = subset.loc[subset['accuracy'].idxmax()]
            
            # cm = best_row['confusion_matrix']

            

            cm = best_row['confusion_matrix']

            if isinstance(cm, str):
                # Remove brackets and convert space-separated values
                cm = cm.replace('[', '').replace(']', '')
                values = np.fromstring(cm, sep=' ')
                cm = values.reshape(2, 2).astype(int)






            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[idx],
                linewidths=0.35,
                cbar=False
            )
            
            axes[idx].set_title(f'{method}\n({best_row["classifier"]})\nAcc: {best_row["accuracy"]:.3f}',
                              fontsize=10)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.suptitle('Confusion Matrices: Best Classifier per Feature Method',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_radar_chart(self, classifier: str = None, save_path: str = None) -> plt.Figure:
        """
        Radar chart comparing metrics across feature methods for a classifier.
        """
        if classifier is None:
            # Find best overall classifier
            classifier = self.results.loc[self.results['accuracy'].idxmax(), 'classifier']
        
        subset = self.results[self.results['classifier'] == classifier]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_mean']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        for idx, (_, row) in enumerate(subset.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['feature_method'], color=self.colors[idx])
            ax.fill(angles, values, alpha=0.1, color=self.colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=10)
        min_val = subset[metrics].min().min()
        max_val = subset[metrics].max().max()
        ax.set_ylim([min_val - 0.005, max_val + 0.005])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title(f'Performance Metrics: {classifier}', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics table.
        """
        summary = self.results.groupby('classifier').agg({
            'accuracy': ['mean', 'std', 'max'],
            'f1_score': ['mean', 'std', 'max'],
            'cv_mean': ['mean', 'std']
        }).round(4)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        
        return summary
    
    def plot_all_visualizations(self, output_dir: str = '../results'):
        """
        Generate all visualizations and save to output directory.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualizations...")
        
        # Accuracy comparison
        self.plot_accuracy_comparison(f'{output_dir}/accuracy_comparison.png')
        print("  - Accuracy comparison saved")
        
        # F1 comparison
        self.plot_f1_comparison(f'{output_dir}/f1_comparison.png')
        print("  - F1 comparison saved")
        
        # CV scores with variance
        self.plot_cv_scores_with_variance(f'{output_dir}/cv_scores.png')
        print("  - CV scores plot saved")
        
        # Heatmaps
        for metric in ['accuracy', 'f1_score', 'cv_mean']:
            self.plot_heatmap(metric, f'{output_dir}/heatmap_{metric}.png')
        print("  - Heatmaps saved")
        
        # Confusion matrices
        self.plot_multiple_confusion_matrices(f'{output_dir}/confusion_matrices.png')
        print("  - Confusion matrices saved")
        
        # Radar charts for top classifiers
        for clf in ['logistic_regression', 'svm', 'naive_bayes']:
            if clf in self.results['classifier'].values:
                self.plot_radar_chart(clf, f'{output_dir}/radar_{clf}.png')
        print("  - Radar charts saved")
        
        # Summary statistics
        summary = self.generate_summary_statistics()
        summary.to_csv(f'{output_dir}/summary_statistics.csv')
        print("  - Summary statistics saved")
        
        print(f"\nAll visualizations saved to {output_dir}/")


def plot_dataset_statistics(df: pd.DataFrame, save_path: str = None):
    """
    Visualize dataset statistics and distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Class distribution
    ax1 = axes[0, 0]
    class_counts = df['label'].value_counts()
    bars = ax1.bar(class_counts.index, class_counts.values, 
                   color=['#3498db', '#e74c3c'], edgecolor='black')
    ax1.set_title('Class Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Samples')
    for bar, count in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', fontsize=11, fontweight='bold')
    
    # Text length distribution
    ax2 = axes[0, 1]
    df['text_length'] = df['text'].str.len()
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        ax2.hist(subset['text_length'], bins=30, alpha=0.6, label=label, edgecolor='black')
    ax2.set_title('Text Length Distribution by Class', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Text Length (characters)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Word count distribution
    ax3 = axes[1, 0]
    df['word_count'] = df['text'].str.split().str.len()
    df.boxplot(column='word_count', by='label', ax=ax3)
    ax3.set_title('Word Count Distribution by Class', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Word Count')
    plt.suptitle('')  # Remove automatic title
    
    # Pie chart
    ax4 = axes[1, 1]
    ax4.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
           colors=['#3498db', '#e74c3c'], explode=[0.02, 0.02],
           shadow=True, startangle=90)
    ax4.set_title('Class Proportion', fontsize=12, fontweight='bold')
    
    plt.suptitle('Dataset Overview and Statistics', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Demo visualization with sample data
    print("Visualization module loaded successfully.")
    print("Use ExperimentVisualizer class for comprehensive experiment visualization.")
