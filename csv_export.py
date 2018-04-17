"""
Export forecasts as computed by model to a .csv file.
"""

from __future__ import division
import fields, model as model
import numpy as np
from collections import defaultdict

    
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
    target = 'revenue'


    names = defaultdict(list)
    add_spec = model.forecast_spec
    for part in np.sort(model.part_components.keys()):
        for c in model.part_components[part]:
            comp_name = part+'comp'+str(c) 
            names[part].append(comp_name) 
            add_spec.append( (comp_name, float, '%6.3f') )

    spec = fields.all_spec

    response, data, weights = \
        model.variables_from_file(input_file, spec=spec, add_spec=add_spec,
                                  logger='error', target=target)

    loadings, components, labels, means, stds = {}, {}, {}, {}, {}
    for part in np.sort(model.part_components.keys()):
        loadings[part], components[part], labels[part], means[part], stds[part] = \
                        model.PLS_fit(response, data, part)
        for i, n in enumerate(names[part]): data[n] = components[part][:, i]
            
    model.add_forecasts(data, response, components, weights)

    data.output(filename=output_file, delimiter=',', print_header=True,
                rename=name_map, exclude=exclude_list)

if __name__ == '__main__':
    main()
    print 'Output completed.'
