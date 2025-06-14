import os
from typing import Dict, Optional
import pandas as pd
import numpy as np


class MetricsLogger():

    def __init__(self, name: str, save_dir: str) -> None:
        self.name = name
        self.save_dir = save_dir + f'/{name}/'
        
        # Dictionary to store metrics
        self.metrics: Dict[str, list] = {}
        self.steps: Dict[str, list] = {}

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, run: Optional[int] = 1) -> None:
        '''
        Log metrics to our custom storage.
        '''
        
        # Store in our custom format
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
                self.steps[metric_name] = []
            self.metrics[metric_name].append(value)
            self.steps[metric_name].append((run, step) if step is not None else (run, len(self.metrics[metric_name])))

    def finalize(self) -> None:
        '''
        Called at the end of training to save all plots and metrics.
        '''
        # self.save_metrics_plot()
        self.save_csv()

    def save_csv(self) -> None:
        '''
        Save metrics to CSV file with proper alignment of steps and values.
        '''
        # First, get all unique steps across all metrics
        all_steps = sorted(set(step for steps in self.steps.values() for step in steps))
        
        # Create a dictionary to store aligned data
        # aligned_data = {'index': all_steps}
        aligned_data = {'run': [t[0] for t in all_steps],
                        'step':[t[1] for t in all_steps]}
        
        # For each metric, create an array of values aligned with all_steps
        for metric_name, metric_values in self.metrics.items():
            # Create a mapping of step to value for this metric
            step_to_value = dict(zip(self.steps[metric_name], self.metrics[metric_name]))
            
            # Create aligned array with NaN for missing steps
            aligned_values = [step_to_value.get(step, np.nan) for step in all_steps]
            aligned_data[metric_name] = aligned_values
            
        
        # Create DataFrame and save
        df = pd.DataFrame(aligned_data)
        csv_path = os.path.join(self.save_dir, 'metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved metrics to {csv_path}')