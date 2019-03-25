import numpy as np
import pandas as pd
import ppgaze

# Prepare your gaze data in advance.
# The data must have information about:
# (i) time stamp
# (ii) x-coordinate of gaze (in pixel)
# (iii) y-coordinate of gaze (in pixel)
# (iv) pupil size.

# Initialize. your_data_xxxx must be an array-like data.
# 'freq' indicates sampling frequency.
data = ppgaze.GazeData(time = times_in_your_data,
                       x = xs_in_your_data, 
                       y = ys_in_your_data, 
                       pupil = pupils_in_your_data,
                       freq = 300)

# Classify data into fixation, saccade, or blink.
# This class has two classification methods: velocity_filter() and dispersion_filter()
# Classified data is saved as data.classified
data.velocity_filter()

# Check the subject fixate on the AOI(s)
# Specify AOIs as dictionary type object.
# Each number indicates [top, right, bottom, left] position ('trouble'!).
# Data is saved as data.gazedata
data.analyze_aoi({'center':[100,100,-100,-100],
               'rightside':[100,400,-100,200]})

# Save preprocessed data as a csv file.
data.gazedata.to_csv('gazedata.csv', index=False, na_rep='NaN)
