from util import datab, logging
import numpy as np
from collections import OrderedDict
import csv
import inspect
import fields

def parse_file(filename, **kwds):

    """Read a csv table into a Datab
    Parameters
    ----------
    filename : string
        The name of the CSV file to loada
    spec : list of tuples
        List of tuples representing all column definitions. Each tuple specifies
        field name, column name or list of column names, data type, format and 
        optional data transformation.
        Data Type Options
            * str, int, float, bool (TODO)
        Format Options
            * 
        Available Cases
            * field name, column name, dtype, format ->
            * field name, column name(s), dtype, format, function ->
    logger : util Logger or string, default 'warning'
        If Logger, logger object to use. If string, specify level of logging/output
        Available Options
            * info ->
            * debug ->
            * warning ->
            * error ->
            * critical ->

    Returns
    -------
    parsed : Datab object
        Datab representing data from the passed in file

    """

    logger = kwds.pop('logger', 'warning')
    if type(logger) == str: logger = logging.Logger('CSVParser', logger)

    spec = kwds.pop('spec', None)

    data = OrderedDict()
    if spec is not None:
        # initialize the data dictionary with column names based on spec
        for field_spec in spec:
            data[field_spec[0]] = []
    else:
        for field_name in reader.fieldnames:
            data[field_name] = []

    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        rownum = 0
        num_expected_columns = len(reader.fieldnames)

        logger.info('Expected number of values in each row: {}'.format(num_expected_columns))
        for row in reader:
            if len(row) <> num_expected_columns:
                logger.warn('Row %d: expecting {} values, but got {}'.format 
                            % (rownum, num_expected_columns, len(row)))
            validate_row(rownum, row, spec, logger, data)
            rownum += 1

    # TODO: this does not currently do any error checking
    # convert spec to datab spec
    db_spec = [(tup[0], tup[2], tup[3]) for tup in spec]
    # TODO: the following may not be the most efficient
    #       look into a different approach for either:
    #           1. Converting dict of columns
    #           2. Using a different data structure during building
    # convert data from dictionary of lists to list of tuples
    # first, transform to list of columns
    list_of_column_lists = [data[key] for key in data.keys()]
    # now, transpose to list of rows
    list_of_row_tuples = [tuple(row) for row in map(None, *list_of_column_lists)]

    return datab.Datab(list_of_row_tuples, spec=db_spec, index='ID', **kwds)



def validate_row(rownum, row, spec, logger, data):
    """Validate row of values based on the spec and append to the data set
    Parameters
    ----------
    rownum : int
        The number of the row within the overall data set
    row : dict
        Dictionary representing the row of data to be validated
    spec : list of tuples
        List of tuples representing all column definitions. Each tuple specifies
        field name, column name or list of column names, data type, format and 
        optional data transformation.
        Data Type Options
            * str, int, float, bool (TODO)
        Format Options
            * 
        Available Cases
            * field name, column name, dtype, format ->
            * field name, column name(s), dtype, format, function ->
    logger : util Logger

    data : OrderedDict
        Dictionary of lists that represent the entire data set. If provided
        the row is validated and all valid data appended to the end of the
        columns.

    Returns
    -------
    is_row_valid : bool
        True if no data issues were encountered or False if problems are found

    """

    logger.info('Validating row %d...' % rownum)
    is_row_valid = True

    if spec is not None:
        calculated_columns = OrderedDict()
        # parse spec for list of columns to load
        for count, field_spec in enumerate(spec):
            value = None
            field_type_str = str(field_spec[2])
            field_type = field_spec[2]
            # more than 4 arguments implies a calculated column
            if len(field_spec) > 4:
                logger.info('Calculating field {}'.format(field_spec[0]))
                calculated_columns[field_spec[0]] = count
                func = field_spec[4]
                # create a list of values based on specified input columns
                try:
                    var_val_list = []
                    for x in field_spec[1]:
                        cur_val = None
                        logger.info('Looking for variable {}'.format(x))
                        # check if the variable is in the current row dictionary
                        if x in row:
                            cur_val = row[x]
                        else:
                            # if the variable is not in the row data set,
                            # it may be in the calculated list 
                            cur_val = data[x][rownum]

                        # convert the current variable to the spec type
                        if 'i' in field_type_str:
                            var_val_list.append(field_type(cur_val))
                        elif 'f' in field_type_str:
                            var_val_list.append(field_type(cur_val))
                        else:
                            var_val_list.append(cur_val)

                    # apply the specified function to the value list
                    logger.info('Applying function to {}'.format(str(var_val_list)))
                    value = func(var_val_list)

                except KeyError, ke:
                    logger.warning('Row %d: column {} not found while calculating \
                                    field {}.'.format(rownum, str(ke), field_spec[0]))
                    is_row_valid = False
                    if 'i' in field_type_str or 'f' in field_type_str:
                        # if the value should be numeric
                        # TODO: Should these be set to float('nan') or None instead?
                        value = 0
                    else:
                        value = None
                except ValueError, ve:
                    logger.warning('Row {}: {}.  Expecting {} for "{}"'.format
                                     (rownum, str(ve), field_type_str, field_spec[0]))
                    is_row_valid = False
                    # this error is thrown if the value is not numeric but should be
                    # TODO: Should these be set to float('nan') or None instead?
                    value = 0

            else:
                logger.info('Looking for field "{}" as "{}".'.format(field_spec[1], field_spec[0]))
                # get the value from the row and convert it to the spec type
                try:
                    if 'i' in field_type_str:
                        value = field_type(row[field_spec[1]])
                    elif 'f' in field_type_str:
                        value = field_type(row[field_spec[1]])
                    else:
                        value = row[field_spec[1]]
                except ValueError, ve:
                    logger.warning('Row {}: {}.  Expecting {} for "{}"'.format
                                    (rownum, str(ve), field_type_str, field_spec[0]))
                    is_row_valid = False
                    if 'i' in field_type_str: value = -1
                    elif 'f' in field_type_str: value = np.nan
                    else: value = None

            logger.info('Adding value "{}" as "{}".'.format(str(value), field_spec[0]))
            data[field_spec[0]].append(value)
        
    else:
        # if no spec is passed, then simply add the row
        for key, val in row:
            try:
                data[key].append(val)
            except KeyError, ke:
                logger.warn('Row {}: Encountered an unexpected column value: {}'.format(rownum, key))
                is_row_valid = False

    return is_row_valid



def main():
    d = parse_file('input.csv', logger='warning', spec=fields.all_spec)
    d.output()


if __name__ == '__main__':
    main()
