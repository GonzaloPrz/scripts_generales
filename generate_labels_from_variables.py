import pandas as pd
from pathlib import Path
import numpy as np

tasks = ['DiaTipico','agradable']
y_labels = ['DASS_21_Depression','DASS_21_Depression_V','Depression_Total_Score','AES_Total_Score',
            'MiniSea_MiniSea_Total_EkmanFaces','MiniSea_minisea_total']

for task in tasks:
    data = pd.read_csv(Path(Path.home(),'data','GeroApathy', f'all_data_{task}.csv'))
    for y_label in y_labels:
        try:
            data[f'{y_label}_label'] = [1 if x >= data[y_label].median() else 0 if not np.isnan(x) else np.nan for x in data[y_label]]
        except:
            pass
    data.to_csv(Path(Path.home(),'data','GeroApathy', f'all_data_{task}.csv'), index=False)