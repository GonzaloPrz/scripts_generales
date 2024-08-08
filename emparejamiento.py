import pandas as pd
import numpy as np
from tableone import TableOne
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import sys

sys.path.append(str(Path(Path.home(),'Proyectos','scripts_generales')))

from matching_module import match

groups = ['AD','PD','FTD']

# Define variables
vars = ['Sexo', 'Edad', 'Escolaridad', 'group', 'Código']
fact_vars = ['Sexo', 'group']
output_var = 'group'

matching_vars = ['Edad','Escolaridad']
cont_vars = ['Edad','Escolaridad']

for group in groups:
    data = pd.read_excel(Path(Path.home(),f'data_{group}.xlsx'))

    data = data.loc[data.Sexo != 3,:].reset_index(drop=True)

    for fact_var in fact_vars:
        data[fact_var] = data[fact_var].astype('category').cat.codes

    matched_data = match(data, fact_vars, cont_vars, output_var, matching_vars)

    matched_data = matched_data.drop_duplicates(subset='Código')

    # Save tables and matched data
    table_before = TableOne(data,list(set(fact_vars + cont_vars) - set(output_var)),fact_vars,groupby=output_var, pval=True, nonnormal=[])

    print(table_before)

    table = TableOne(matched_data,list(set(fact_vars + cont_vars) - set(output_var)),fact_vars,groupby=output_var, pval=True, nonnormal=[])

    print(table)

    matched_data.to_excel(Path(Path.home(),f'data_matched_{group}.xlsx'), index=False)
    table_before.to_excel(Path(Path.home(),f'table_before_{group}.xlsx'))
    table.to_excel(Path(Path.home(),f'table_matched_{group}.xlsx'))