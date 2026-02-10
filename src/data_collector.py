"""
Data Collection Module for Sports vs Politics Classification
=============================================================
This module handles data collection from multiple sources and provides
utilities for dataset creation and management.

Data Sources:
1. BBC News Dataset (Sports & Politics categories)
2. Synthetic data generation for testing
3. Recent Indian news articles via NewsAPI.org
By:
Priyadip Sau (M25CSA023)
"""

import numpy as np
import pandas as pd
import os
import random
from typing import Dict
import requests
import re
import csv



# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

# Vocabulary and templates for synthetic data generation
SPORTS_VOCABULARY = {
    'nouns': [
        'team', 'player', 'coach', 'match', 'game', 'tournament', 'championship',
        'league', 'season', 'victory', 'defeat', 'goal', 'point', 'score',
        'athlete', 'stadium', 'arena', 'trophy', 'medal', 'record', 'performance',
        'training', 'competition', 'finals', 'playoffs', 'World Cup', 'Olympics',
        'quarterback', 'striker', 'defender', 'goalkeeper', 'pitcher', 'batter'
    ],
    'verbs': [
        'won', 'lost', 'scored', 'defeated', 'played', 'competed', 'trained',
        'signed', 'transferred', 'retired', 'injured', 'recovered', 'dominated',
        'secured', 'celebrated', 'advanced', 'eliminated', 'qualified', 'finished'
    ],
    'adjectives': [
        'talented', 'skilled', 'experienced', 'young', 'veteran', 'promising',
        'dominant', 'impressive', 'outstanding', 'remarkable', 'historic',
        'competitive', 'professional', 'amateur', 'international', 'national'
    ],
    'sports': [
        'football', 'basketball', 'soccer', 'baseball', 'tennis', 'golf',
        'hockey', 'cricket', 'rugby', 'swimming', 'athletics', 'boxing',
        'cycling', 'volleyball', 'wrestling', 'Formula 1', 'motorsport'
    ],
    'teams': [
        'Manchester United', 'Real Madrid', 'Lakers', 'Patriots', 'Yankees',
        'Barcelona', 'Bayern Munich', 'Chicago Bulls', 'Dallas Cowboys',
        'Liverpool', 'Arsenal', 'Golden State Warriors', 'New England Patriots'
    ],
    'names': [
        'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Martinez',
        'Anderson', 'Thompson', 'White', 'Harris', 'Robinson', 'Clark', 'Lewis'
    ]
}

POLITICS_VOCABULARY = {
    'nouns': [
        'government', 'parliament', 'congress', 'senate', 'president', 'minister',
        'policy', 'legislation', 'bill', 'law', 'election', 'vote', 'campaign',
        'party', 'coalition', 'opposition', 'budget', 'reform', 'debate',
        'democracy', 'constitution', 'diplomat', 'ambassador', 'treaty', 'summit',
        'administration', 'cabinet', 'committee', 'referendum', 'poll'
    ],
    'verbs': [
        'announced', 'proposed', 'passed', 'vetoed', 'debated', 'approved',
        'rejected', 'signed', 'implemented', 'reformed', 'criticized', 'supported',
        'condemned', 'negotiated', 'addressed', 'advocated', 'campaigned', 'voted'
    ],
    'adjectives': [
        'political', 'bipartisan', 'controversial', 'unprecedented', 'historic',
        'economic', 'foreign', 'domestic', 'federal', 'legislative', 'executive',
        'democratic', 'republican', 'conservative', 'liberal', 'progressive'
    ],
    'topics': [
        'healthcare', 'education', 'immigration', 'taxation', 'climate change',
        'national security', 'foreign policy', 'trade', 'infrastructure',
        'social security', 'defense', 'economy', 'unemployment', 'inflation'
    ],
    'institutions': [
        'White House', 'Capitol Hill', 'Supreme Court', 'Pentagon', 'State Department',
        'United Nations', 'European Union', 'NATO', 'World Bank', 'IMF',
        'House of Representatives', 'Senate', 'Federal Reserve'
    ],
    'names': [
        'Senator', 'Representative', 'Secretary', 'Governor', 'Mayor',
        'Chancellor', 'Prime Minister', 'President', 'Speaker', 'Chairman'
    ]
}

SPORTS_TEMPLATES = [
    "The {team} {verb} their opponents in a thrilling {sport} {noun} yesterday.",
    "{name} {verb} an incredible performance, leading the {team} to {noun}.",
    "In a {adjective} display of {sport}, the {team} secured their place in the {noun}.",
    "The {adjective} {noun} between {team} ended with a {adjective} {noun}.",
    "Coach {name} praised the team's {adjective} performance after the {noun}.",
    "The {sport} {noun} saw {team} deliver a {adjective} victory against fierce competition.",
    "{name}, the {adjective} {noun}, {verb} three times during the intense {noun}.",
    "Fans celebrated as {team} {verb} their rivals in the championship {noun}.",
    "The {adjective} {sport} season continues with {team} maintaining their dominant form.",
    "Breaking: {name} {verb} a record-breaking contract with {team} worth millions.",
    "The {noun} witnessed an extraordinary comeback as {team} {verb} in the final minutes.",
    "Sports analysts praised the {adjective} tactics employed by {team} throughout the {noun}.",
    "The {sport} community reacted to {name}'s {adjective} announcement about retirement.",
    "{team}'s {adjective} run in the {noun} continues despite mounting injuries.",
    "The international {sport} {noun} kicks off with {team} as favorites to win."
]

POLITICS_TEMPLATES = [
    "The {institution} {verb} new {topic} {noun} amid growing concerns.",
    "{title} {name} {verb} the controversial {topic} {noun} during the session.",
    "In a {adjective} move, the {institution} {verb} comprehensive {topic} reforms.",
    "The {adjective} {noun} on {topic} sparked intense {noun} across party lines.",
    "Critics {verb} the administration's approach to {topic} as {adjective}.",
    "The {noun} over {topic} continues as lawmakers prepare for crucial votes.",
    "{title} {name} addressed the nation regarding the {adjective} {topic} situation.",
    "Bipartisan support emerged for the {adjective} {topic} {noun} proposed last week.",
    "The {institution} faces mounting pressure to address {topic} concerns.",
    "Political analysts describe the {noun} as a {adjective} moment for {topic} policy.",
    "Negotiations between parties {verb} progress on the {topic} {noun}.",
    "The administration's {topic} agenda faces {adjective} opposition from lawmakers.",
    "Voters expressed concerns about {topic} during the {adjective} town hall meeting.",
    "The {adjective} {noun} highlighted deep divisions on {topic} within the {institution}.",
    "International observers {verb} the country's {topic} {noun} with interest."
]


def fill_template(template: str, vocabulary: dict, extra_fields: dict = None) -> str:
    """Fill a template with random vocabulary selections."""
    filled = template
    
    # Standard replacements
    field_mapping = {
        '{noun}': 'nouns',
        '{verb}': 'verbs',
        '{adjective}': 'adjectives',
        '{sport}': 'sports',
        '{team}': 'teams',
        '{topic}': 'topics',
        '{institution}': 'institutions'
    }
    
    for placeholder, key in field_mapping.items():
        while placeholder in filled:
            if key in vocabulary:
                filled = filled.replace(placeholder, random.choice(vocabulary[key]), 1)
            else:
                break
    
    # Handle names
    while '{name}' in filled:
        filled = filled.replace('{name}', random.choice(vocabulary.get('names', ['Smith'])), 1)
    
    # Handle titles for politics
    while '{title}' in filled:
        filled = filled.replace('{title}', random.choice(
            vocabulary.get('names', ['Senator', 'Representative'])
        ), 1)
    
    return filled


def generate_sports_article(min_sentences: int = 3, max_sentences: int = 8) -> str:
    """Generate a synthetic sports article."""
    num_sentences = random.randint(min_sentences, max_sentences)
    sentences = []
    
    for _ in range(num_sentences):
        template = random.choice(SPORTS_TEMPLATES)
        sentence = fill_template(template, SPORTS_VOCABULARY)
        sentences.append(sentence)
    
    return ' '.join(sentences)


def generate_politics_article(min_sentences: int = 3, max_sentences: int = 8) -> str:
    """Generate a synthetic politics article."""
    num_sentences = random.randint(min_sentences, max_sentences)
    sentences = []
    
    for _ in range(num_sentences):
        template = random.choice(POLITICS_TEMPLATES)
        sentence = fill_template(template, POLITICS_VOCABULARY)
        sentences.append(sentence)
    
    return ' '.join(sentences)


def generate_synthetic_dataset(n_samples: int = 2000, 
                               balanced: bool = True,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset for sports vs politics classification.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    balanced : bool
        If True, generate equal samples for each class
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'text' and 'label' columns
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data = []
    
    if balanced:
        n_per_class = n_samples // 2
        
        # Generate sports articles
        for i in range(n_per_class):
            text = generate_sports_article()
            data.append({'text': text, 'label': 'sports', 'id': f'sports_{i}'})
        
        # Generate politics articles
        for i in range(n_per_class):
            text = generate_politics_article()
            data.append({'text': text, 'label': 'politics', 'id': f'politics_{i}'})
    else:
        # Unbalanced dataset
        sports_ratio = random.uniform(0.3, 0.7)
        n_sports = int(n_samples * sports_ratio)
        n_politics = n_samples - n_sports
        
        for i in range(n_sports):
            text = generate_sports_article()
            data.append({'text': text, 'label': 'sports', 'id': f'sports_{i}'})
        
        for i in range(n_politics):
            text = generate_politics_article()
            data.append({'text': text, 'label': 'politics', 'id': f'politics_{i}'})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
    
    return df


# =============================================================================
# REAL DATA LOADING UTILITIES
# ============================================================================


def load_bbc_csv(csv_path):
    df = pd.read_csv(csv_path, sep='\t', encoding='latin1')


    # Clean column names (VERY IMPORTANT)
    df.columns = df.columns.str.strip().str.lower()

    print("Columns found:", df.columns)

    # Now filter
    df = df[df['category'].isin(['sport', 'politics'])].copy()


    # # Normalize column names (in case slight differences)
    # df.columns = [c.strip() for c in df.columns]

    # # Keep only relevant categories
    # df = df[df['category'].isin(['sport', 'politics'])].copy()

    # Ensure required columns exist
    for col in ['filename', 'title', 'content']:
        if col not in df.columns:
            df[col] = ''

    # Add Source / Type
    df['Source'] = 'BBC'
    df['Type'] = 'real'

    # Reorder columns exactly as requested
    df = df[['Source', 'Type', 'category', 'filename', 'title', 'content']].reset_index(drop=True)
    return df



def clean_api_news(text):
    if not text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove truncation markers like [+3949 chars]
    text = re.sub(r'\[\+\d+\s*chars\]', '', text)

    # Remove common website junk
    junk_patterns = [
        r'Follow Us.*',
        r'India News.*',
        r'Breaking News.*',
        r'Click here.*',
        r'Read more.*'
    ]

    for p in junk_patterns:
        text = re.sub(p, '', text, flags=re.IGNORECASE)

    # Remove quotes
    text = text.replace('"', '').replace("'", "")

    # Remove newlines/tabs
    text = re.sub(r'[\n\r\t]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()



def load_recent_indian_news(api_key: str, max_per_category: int = 100) -> pd.DataFrame:
    """
    Fetch recent Indian news using NewsAPI.org.
    Returns DataFrame with columns: Source, Type, category, filename, title, content
    Categories fetched: 'sports' -> label 'sport', 'general' -> label 'politics'
    """
    base_url = "https://newsapi.org/v2/everything"

    rows = []
    print("Fetching news from NewsAPI...",api_key)

    def fetch_and_normalize(category_param, label):

        # Multiple queries per category
        if category_param == "sports":
            queries = [
                "india cricket",
                "indian premier league",
                "virat kohli",
                "rohit sharma",
                "india football",
                "indian super league",
                "india olympics",
                "badminton india",
                "pv sindhu",
                "kabaddi india",
                "pro kabaddi league",
                "hockey india",
                "indian hockey team",
                "wrestling india",
                "boxing india",
                "athletics india",
                "tennis india",
                "neeraj chopra",
                "ipl cricket",
                "t20 world cup india",
                "india test cricket",
                "sports authority india",
                "khelo india",
                "mohun bagan",
                "jhulan goswami cricket",
                "kolkata knight riders",
                "asian games india",
                "indian football derby match",
                "mohammedan sporting match",
                "east bengal club match",
                "ipl knight riders match",
                "eden gardens cricket match",
                "commonwealth games india",
                "wriddhiman saha cricket"
                ]

        else:
            queries = [
                "india election",
                "indian parliament",
                "india government",
                "narendra modi",
                "bjp india",
                "congress party india",
                "rahul gandhi",
                "mamata banerjee",
                "amit shah",
                "indian politics",
                "lok sabha",
                "rajya sabha",
                "india foreign policy",
                "india budget",
                "indian economy policy",
                "supreme court india politics",
                "india diplomacy",
                "india cabinet meeting",
                "delhi government",
                "indian constitution",
                "public policy india",
                "india defence policy",
                "indian political parties",
                "india governance",
                "india political debate",
                "trinamool congress",
                "kolkata political protest",
                "bengal election commission",
                "west bengal cabinet"
            ]

        for query in queries:

            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "apiKey": api_key
            }

            resp = requests.get(base_url, params=params, timeout=20)

            print(f"Status ({query}):", resp.status_code)

            if resp.status_code != 200:
                print("NewsAPI error:", resp.text)
                continue

            articles = resp.json().get("articles", [])

            for i, a in enumerate(articles):
                title = a.get("title") or ""
                description = a.get("description") or ""
                content = a.get("content") or ""

                combined = clean_api_news(title + " " + description + " " + content)

                if len(combined.split()) < 8:
                    continue

                rows.append({
                    "Source": "NewsAPI",
                    "Type": "api",
                    "category": label,
                    "filename": f"api_{category_param}_{query}_{i}",
                    "title": title,
                    "content": combined
                })



    fetch_and_normalize("sports", "sport")
    fetch_and_normalize("general", "politics")

    # return pd.DataFrame(rows)[['Source','Type','category','filename','title','content']]
    if not rows:
        print(" No NewsAPI data fetched.")
        return pd.DataFrame(columns=[
            'Source','Type','category','filename','title','content'
        ])

    df = pd.DataFrame(rows)
    return df[['Source','Type','category','filename','title','content']]




# =============================================================================
# DATASET ANALYSIS UTILITIES
# =============================================================================

class DatasetAnalyzer:
    """Comprehensive dataset analysis and statistics."""
    
    def __init__(self, df: pd.DataFrame, text_col: str = 'text', 
                 label_col: str = 'label'):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        
    def basic_statistics(self) -> Dict:
        """Calculate basic dataset statistics."""
        stats = {
            'total_samples': len(self.df),
            'class_distribution': self.df[self.label_col].value_counts().to_dict(),
            'class_percentage': (self.df[self.label_col].value_counts(normalize=True) * 100).to_dict()
        }
        
        # Text length statistics
        self.df['text_length'] = self.df[self.text_col].str.len()
        self.df['word_count'] = self.df[self.text_col].str.split().str.len()
        
        stats['avg_text_length'] = self.df['text_length'].mean()
        stats['std_text_length'] = self.df['text_length'].std()
        stats['avg_word_count'] = self.df['word_count'].mean()
        stats['std_word_count'] = self.df['word_count'].std()
        
        # Per-class statistics
        stats['per_class_stats'] = {}
        for label in self.df[self.label_col].unique():
            subset = self.df[self.df[self.label_col] == label]
            stats['per_class_stats'][label] = {
                'count': len(subset),
                'avg_text_length': subset['text_length'].mean(),
                'avg_word_count': subset['word_count'].mean(),
                'min_word_count': subset['word_count'].min(),
                'max_word_count': subset['word_count'].max()
            }
        
        return stats
    
    def vocabulary_analysis(self, top_n: int = 20) -> Dict:
        from collections import Counter
        import re
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))

        results = {}

        for label in self.df[self.label_col].unique():
            subset = self.df[self.df[self.label_col] == label]
            all_text = ' '.join(subset[self.text_col].tolist())

            # Tokenization
            words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

            # Remove stopwords
            words = [w for w in words if w not in stop_words and len(w) > 2]

            word_freq = Counter(words)

            results[label] = {
                'total_words': len(words),
                'unique_words': len(word_freq),
                'top_words': word_freq.most_common(top_n),
                'vocabulary_richness': len(word_freq) / len(words) if words else 0
            }

        return results

    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        stats = self.basic_statistics()
        vocab = self.vocabulary_analysis()
        
        report = []
        report.append("=" * 70)
        report.append("DATASET ANALYSIS REPORT")
        report.append("=" * 70)
        
        report.append(f"\nTotal Samples: {stats['total_samples']}")
        report.append("\nClass Distribution:")
        for label, count in stats['class_distribution'].items():
            pct = stats['class_percentage'][label]
            report.append(f"  {label}: {count} ({pct:.1f}%)")
        
        report.append(f"\nText Statistics:")
        report.append(f"  Average text length: {stats['avg_text_length']:.1f} characters")
        report.append(f"  Average word count: {stats['avg_word_count']:.1f} words")
        
        report.append("\nPer-Class Statistics:")
        for label, cls_stats in stats['per_class_stats'].items():
            report.append(f"\n  {label.upper()}:")
            report.append(f"    Samples: {cls_stats['count']}")
            report.append(f"    Avg words: {cls_stats['avg_word_count']:.1f}")
            report.append(f"    Word range: {cls_stats['min_word_count']} - {cls_stats['max_word_count']}")
        
        report.append("\nVocabulary Analysis:")
        for label, vocab_stats in vocab.items():
            report.append(f"\n  {label.upper()}:")
            report.append(f"    Total words: {vocab_stats['total_words']}")
            report.append(f"    Unique words: {vocab_stats['unique_words']}")
            report.append(f"    Vocabulary richness: {vocab_stats['vocabulary_richness']:.4f}")
            report.append(f"    Top 10 words: {[w for w, _ in vocab_stats['top_words'][:10]]}")
        
        return '\n'.join(report)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Collect BBC CSV, synthetic, and recent Indian news; save final CSV."""
    print("Starting data collection pipeline...")

    # 1) Load BBC CSV file (update path if needed)
    bbc_csv_path = "../data/bbc-news-data.csv"   
    if os.path.exists(bbc_csv_path):
        print("Loading BBC CSV:", bbc_csv_path)
        df_bbc = load_bbc_csv(bbc_csv_path)
        print("  BBC rows:", len(df_bbc))
    else:
        print("BBC CSV not found at", bbc_csv_path)
        df_bbc = pd.DataFrame(columns=['Source','Type','category','filename','title','content'])

    # 2) Synthetic data
    print("Generating synthetic data...")
    df_syn = generate_synthetic_dataset(n_samples=1000, balanced=True, seed=42)
    df_syn['Source'] = 'Synthetic'
    df_syn['Type'] = 'synthetic'
    # map labels to match category names (sport / politics)
    df_syn['category'] = df_syn['label'].map({'sports':'sport','politics':'politics'})
    df_syn['filename'] = df_syn['id']
    df_syn['title'] = 'Synthetic Article'
    df_syn['content'] = df_syn['text']
    df_syn = df_syn[['Source','Type','category','filename','title','content']]
    print("  Synthetic rows:", len(df_syn))

    # 3) Recent Indian news via NewsAPI
    API_KEY = "ab61c93bbc3a4568824c1357e1de087a"  # <- replace with your key
    if API_KEY == "YOUR_NEWSAPI_KEY_HERE":
        print("Warning: NewsAPI API key not set. Skipping NewsAPI fetch.")
        df_api = pd.DataFrame(columns=['Source','Type','category','filename','title','content'])
    else:
        print("Fetching recent Indian news via NewsAPI...")
        df_api = load_recent_indian_news(API_KEY, max_per_category=50)
        print("  NewsAPI rows:", len(df_api))

    # 4) Combine datasets
    df_final = pd.concat([df_bbc, df_syn, df_api], ignore_index=True, sort=False)
    # Ensure columns and order
    df_final = df_final[['Source','Type','category','filename','title','content']]
    df_final['content'] = df_final['content'].str.replace('\n',' ')
    df_final['title'] = df_final['title'].str.replace('\n',' ')

    # -----------------------------------------------------
    # Dataset Analysis (IMPORTANT FOR REPORT)
    # -----------------------------------------------------

    # Create temporary text column for analysis
    df_analysis = df_final.copy()
    df_analysis['text'] = df_analysis['title'].fillna('') + " " + df_analysis['content'].fillna('')
    df_analysis['label'] = df_analysis['category']

    analyzer = DatasetAnalyzer(df_analysis, text_col='text', label_col='label')
    report = analyzer.generate_report()

    print("\nDATASET ANALYSIS REPORT")
    print(report)

    # Save analysis report to file
    os.makedirs('../data', exist_ok=True)
    with open('../data/dataset_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 5) Shuffle and save dataset
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    out_path = '../data/final_dataset.csv'

    df_final.to_csv(
        out_path,
        index=False,
        encoding='utf-8',
        quoting=csv.QUOTE_ALL
        )
    print(f"Final dataset saved to: {out_path}  (rows: {len(df_final)})")
    return df_final
if __name__ == "__main__":
    df = main()
