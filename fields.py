from __future__ import division
import numpy as np


target_spec = \
[  ('ID', 'Firm ID', np.int64, '%4d')
   , ('target_revenue', ['Total revenue - 2017'],
      np.float32, '%5.1f', lambda x: x[0]/1000)
]

physical_spec = \
[  ('ID', 'Firm ID', np.int64, '%4d')
   , ('age', 'Firm age', np.int64, '%2d')
   , ('size', 'Firm size', np.float32, '%4.0f')
   , ('floors', 'Firm # floors', np.int64, '%2d')   
   , ('staff_num', 'Firm avg # staff', np.float32, '%4.2f')
   , ('manager_num', 'Firm # manager', np.float32, '%4.1f')
]

all_spec = target_spec + \
            physical_spec[1:]