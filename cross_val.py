"""
Cross validate PLS model.
"""
from __future__ import division
import model as model
import numpy as np
from util import regress

def cross_validate(rw, data, weights, test_data_ratio, num_iterations,
                   parts=None, merged_test=False, coefs_test=False,
                   forecast=False, datab=None):
    """
    coefs_test:
    Run regression all the components individually.
    
    merged_test:
    Concatenate all datapoints across iterations and run one regression for each
    of train and test data.
  """

    if datab is None: datab = not forecast
    include_parts = model.part_components.keys()

    if parts is not None:
        include_parts = list(set(include_parts) & set(parts))

    names_list = []
    names = {}
    for part in include_parts:
        names[part] = [part+'comp'+str(c) for c in model.part_components[part]]
        names_list = names_list + names[part]

    test_names = ['const'] + (names_list if coefs_test else ['forecast'])

    # The following two will contain results per iteration, and will be returned.
    train_results, test_results = [], []
    if merged_test:
        train_regress = regress.Regress(npreds=len(names_list) if coefs_test else 1,
                                        weighted=True, constant=True, store_all=True)
        test_regress = regress.Regress(npreds=len(names_list) if coefs_test else 1,
                                       weighted=True, constant=True, store_all=True)
        
    overlay = np.ones(len(rw), dtype=bool)
    num_test = int(round(test_data_ratio * len(rw)))
    overlay[-num_test:] = False  # select all but last n as train data
    # Here's where we start looping, for N number of iterations
    for i in range(num_iterations):

        np.random.shuffle(overlay)

        loadings, components, labels, means, stds = {}, {}, {}, {}, {}
        for part in np.sort(include_parts):
            loadings[part], components[part], labels[part], means[part], stds[part] = \
                        model.PLS_fit(rw[overlay], data[overlay], part)

        predictors = np.hstack(tuple(components[c] for c in include_parts))
        if (not merged_test) or (not coefs_test):
            # Fit train data forecast with above PLS components.
            train_result = regress.regress(rw[overlay], predictors,
                                           weights[overlay], datab=False, names=names_list,
                                           forecast=merged_test and not coefs_test)
            train_resultD = regress.Datab([train_result])
            
        if merged_test:
            train_regress.update(rw[overlay], predictors if coefs_test else
                                 train_result['forecast'], weights[overlay])
        else: train_results.append(train_result)

        # Now on the test data. Notice ~overlay rather than overlay.
        # --------------------

        # Obtain PLS components for test data, using the same fit as for the train data.
        for part in include_parts:
            component, dummy = model.PLS_X(data[~overlay], part)
            component = component - means[part][np.newaxis, :]
            component = component / stds[part][np.newaxis, :]
            components[part] = np.mat(component) * np.mat(loadings[part])

        if coefs_test:
            predictors = np.hstack(tuple(components[c] for c in include_parts))
        else:
            # Obtain forecast for test data, using the same regression coefficients obtained
            # for the train data, on the above PLS components.
            predictors = np.zeros(len(rw[~overlay])) + train_resultD['const']
            for part in include_parts:
                predictors += np.ravel(components[part] * np.array([train_resultD[n]
                                                                    for n in names[part]]))
            
        if merged_test:
            test_regress.update(rw[~overlay], predictors, weights[~overlay])
        else:
            test_results.append(regress.regress(rw[~overlay], predictors, weights[~overlay],
                                                constant=False, datab=False, names=test_names))

    if merged_test:
        test_results = test_regress.compute(redundancy=num_iterations * test_data_ratio,
                                            forecast=forecast, errors=forecast)
        train_results = train_regress.compute(redundancy=num_iterations * (1 - test_data_ratio),
                                              forecast=forecast, errors=forecast)
        if forecast:
            train_results['response'] = np.concatenate([u[0] for u in
                                                        train_regress.Multivariate.all_update])[:, -1]
            train_results['weights'] = np.concatenate([u[1] for u in
                                                       train_regress.Multivariate.all_update])
            test_results['response'] = np.concatenate([u[0] for u in
                                                        test_regress.Multivariate.all_update])[:, -1]
            test_results['weights'] = np.concatenate([u[1] for u in
                                                       test_regress.Multivariate.all_update])
    
    if datab:
        return regress.Datab(train_results, formats='%6.3f', names=test_names), \
               regress.Datab(test_results, formats='%6.3f', names=test_names)
    else:
        return train_results, test_results


def main():

    test_data_ratio = 0.1
    num_iterations = 100
    parts = "A"
    #coefs_test = True
    
    rw, data, weights = model.variables_from_file('input.csv', logger='error')

    train, test = cross_validate(rw, data, weights,
                                 test_data_ratio, num_iterations,
                                 parts=parts.split(',')
                                 , merged_test=True
                                 , coefs_test=True
                                 , forecast=False
                                 )
   
    print '\nTraining data\n', '=============\n'
    train.output(exclude=['d_count', 'mean_wt'])
    print '\nTest data\n', '=========\n'
    test.output(exclude=['d_count', 'mean_wt'])


    train, test = cross_validate(rw, data, weights,
                                 test_data_ratio, num_iterations,
                                 parts=parts.split(',')
                                 , merged_test=True
                                 , coefs_test=False
                                 , forecast=False
                                 )   
    print '\nTraining data\n', '=============\n'
    train.output(exclude=['d_count', 'mean_wt'])
    print '\nTest data\n', '=========\n'
    test.output(exclude=['d_count', 'mean_wt'])

if __name__ == '__main__':
    main()
