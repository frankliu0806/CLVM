"""
Model spreadsheet data.
"""
from __future__ import division
import numpy as np
from util.arrays import per_cap
from util import regress
from sklearn.cross_decomposition import PLSRegression
import pandas

# helper function used to cap bottom and top 2 branches. (i.e. 1.5%le)
pc = lambda x :  per_cap(x, [1.5, 98.5])

# Which PLS components to take for each part of the modeling.
part_components = {'A': (0, 1, 2,)}


def PLS_X(data, part):
    """
    Return X for PLS regression, along with column names.

    """
        
    if part == 'A':
        return np.array([data['Firm age'],
                         pc(data['Firm size']),
                         pc(data['Firm # floors']),
                         data['Firm # manager'],
                         data['Firm avg # staff']]).T, \
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

    regress_func = regress.panel_summary
    
    df = pandas.read_csv(input_file)
    response = df["Total revenue - 2017"]
    data = df
    weights = [1] * len(df)
    
    loadings, components, labels, names, means, stds = {}, {}, {}, {}, {}, {}
    
    for part in np.sort(part_components.keys()):
        
        loadings[part], components[part], labels[part], means[part], stds[part] = \
                        PLS_fit(response, data, part) 

        names[part] = [part+'comp'+str(c) for c in part_components[part]]         

        #Print_loadings
        print("loadings")
        print("======================================")
        print_loadings(names[part], labels[part], loadings[part])
        
        print("\nscores/coefficients")
        print("======================================")        

        regress_func(response, components[part], weights=None, formats='%7.4f',
                     exclude=['d_count', 'mean_wt'], names=names[part])       
        
            
if __name__ == '__main__':
    main()
