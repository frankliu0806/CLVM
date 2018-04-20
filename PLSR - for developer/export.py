"""
Export forecasts as computed by model to a .csv file.
"""

from __future__ import division
import model as model
import numpy as np
from util import regress
from collections import defaultdict
import os, csv
import pandas
    
# List of columns to rename in the resulting CSV file
# Each tuple corresponds to a single column to rename in the format of:
#       ('data column name', 'output column name')
name_map = \
    []

# List of columns to exclude from the output CSV file
exclude_list = \
    []



def main():

    input_file = 'input.csv'
    output_file = 'output.csv'

    names = defaultdict(list)

    
    for part in np.sort(model.part_components.keys()):
        for c in model.part_components[part]:
            comp_name = part+'comp'+str(c) 
            names[part].append(comp_name) 


    df = pandas.read_csv(input_file)
    response = df["Total revenue - 2017"]
    data = df
    weights = [1] * len(df)

    loadings, components, labels, means, stds = {}, {}, {}, {}, {}
    for part in np.sort(model.part_components.keys()):
        loadings[part], components[part], labels[part], means[part], stds[part] = \
                        model.PLS_fit(response, data, part)
        for i, n in enumerate(names[part]): data[n] = components[part][:, i]
            
    model.add_forecasts(data, response, components, weights)
    data.to_csv(output_file)


if __name__ == '__main__':
    main()
    print 'Output completed.'
