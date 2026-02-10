import pandas as pd
from visualization import ExperimentVisualizer

results = pd.read_csv('../results/experiment_results.csv')
viz = ExperimentVisualizer(results)
viz.plot_all_visualizations('../results')
