import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.multicomp import (pairwise_tukeyhsd)


data = np.rec.array([
(  1,   'mental',  2 ),
(  2,   'mental',  2 ),
(  3,   'mental',  3 ),
(  4,   'mental',  4 ),
(  5,   'mental',  4 ),
(  6,   'mental',  5 ),
(  7,   'mental',  3 ),
(  8,   'mental',  4 ),
(  9,   'mental',  4 ),
( 10,   'mental',  4 ),
( 11, 'physical',  4 ),
( 12, 'physical',  4 ),
( 13, 'physical',  3 ),
( 14, 'physical',  5 ),
( 15, 'physical',  4 ),
( 16, 'physical',  1 ),
( 17, 'physical',  1 ),
( 18, 'physical',  2 ),
( 19, 'physical',  3 ),
( 20, 'physical',  3 ),
( 21,  'medical',  1 ),
( 22,  'medical',  2 ),
( 23,  'medical',  2 ),
( 24,  'medical',  2 ),
( 25,  'medical',  3 ),
( 26,  'medical',  2 ),
( 27,  'medical',  3 ),
( 28,  'medical',  1 ),
( 29,  'medical',  3 ),
( 30,  'medical',  1 )], dtype=[('idx', '<i4'),
                                ('Treatment', '|S8'),
                                ('StressReduction', '<i4')])

df = pd.DataFrame(data)


#One way ANOVA
grps = pd.unique(df.Treatment.values)
group_data = {grp:df['StressReduction'][df.Treatment == grp] for grp in grps}

F, p = stats.f_oneway(group_data['mental'], group_data['physical'], group_data['medical'])
print("F = %f" %F)
print("p = %f" %p)

#Tukey's HSD
result = pairwise_tukeyhsd(df['StressReduction'], df['Treatment'])
print("\n")
print result




