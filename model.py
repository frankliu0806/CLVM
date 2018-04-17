"""
Model spreadsheet data.
"""
from __future__ import division
import fields, read_csv
import numpy as np
from util.arrays import per_cap
from util import regress
from sklearn.cross_decomposition import PLSRegression
from numpy.ma.core import nomask


# helper function used to cap bottom and top 2 branches. (i.e. 1.5%le)
pc = lambda x :  per_cap(x, [1.5, 98.5])

# Which PLS components to take for each part of the modeling.

part_components = {'A': (0, 1, 2,)}

forecast_spec = [('forecast', float, '%6.3f'),]


def variables_from_file(filename, target='revenue', spec=fields.all_spec, logger='warning',
                        add_spec=forecast_spec):
    """
    Read data from given .csv file, returning target to model, the predictors
    and weights.

    target:
    Look for field 'target_' + <target> in spec or fields.target_spec, returning
    values found there after normalizing, as the target.
    
    skip_new_branches:
    Remove rows for branches with ID in new_branch_IDs.
    """

    data = read_csv.parse_file(filename, spec=spec, add_spec=add_spec, logger=logger)
    
    target_field = 'target_' + target
    target_data = data

    unit_weight = [1] * len(target_data)  
    weights = np.ma.array(unit_weight, mask=nomask) 
    
    response = np.ma.array(target_data[target_field], mask=nomask)
    response /= np.average(response, weights=weights)
    response -= 1.0
    
    return response, data, weights


def PLS_X(data, part):
    """
    Return X for PLS regression, along with column names.

    """
        
    if part == 'A':
        return np.array([data['age'], pc(data['size']), pc(data['floors']), data['staff_num'], data['manager_num']]).T, \
        ['age', 'size', 'floors', 'staffN','managerN']



def PLS_fit(response, data, part):
    """
    Perform PLS regression, returning
    a) loading for each of the X columns for each of the components
    b) the component values themselves
    c) the labels for the X columns

    """
    components = part_components[part]
    n_components = components[-1] + 1 if type(components) == tuple else components
    pls = PLSRegression(n_components=n_components) 
    
    X, xlabels = PLS_X(data, part)
    pls.fit(X, response)

    loadings = pls.x_rotations_[:, components]
    scores = pls.x_scores_[:, components]
    
    return loadings, scores, xlabels, pls.x_mean_, pls.x_std_


def PLS_comps(response, data, parts):
    """Run PLS_fit for each part, returning the hstack of all the components."""

    components = []
    for part in parts:
        ldings, comps, lbls, mns, sts = PLS_fit(response, data, part)
        if not len(components): components = comps
        else: components = np.hstack((components, comps))
        
    return components

        
def print_loadings(names, labels, loadings):
    """Print nice columns of loadings."""
    
    print '\n          ' + ' '.join('%6s' % n for n in names)
    for label, loading in zip(labels, loadings):
        print '%-8s ' % label,
        print ' '.join('%6.3f' % l for l in loading)
    print ''


def get_forecast(response, components, weights):
    """
    Return forecast based on regressing response against components with given weights.
    """

    results = regress.regress(response, np.hstack(components),
                              weights, forecast=True, datab=False)
    return results['forecast']
    
    
def add_forecasts(data, response, components, weights, target='revenue'):
    
    data['forecast'] = get_forecast(response, components['A'], weights)

def main():

    input_file= 'input.csv' 
    target = 'revenue'
    
    regress_func = regress.panel_summary
    
    spec = fields.all_spec
    
    response, data, weights = \
              variables_from_file(input_file, target=target,
                                  spec=spec, logger='error')        

    
    loadings, components, labels, names, means, stds = {}, {}, {}, {}, {}, {}

    
    for part in np.sort(part_components.keys()):
        
        loadings[part], components[part], labels[part], means[part], stds[part] = \
                        PLS_fit(response, data, part) 

        names[part] = ['comp'+str(c) for c in part_components[part]]         

        #Print_loadings
        print("loadings")
        print("======================================")
        print_loadings(names[part], labels[part], loadings[part])
        
        print("\nscores/coefficients")
        print("======================================")        

        regress_func(response, components[part], weights, formats='%7.4f',
                     exclude=['d_count', 'mean_wt'], names=names[part])       
       

    
    
if __name__ == '__main__':
    main()
