import copy
import math
import numpy as np
import pandas as pd

class GazeData:

    def __init__(self, time, x, y, pupil, freq=600,
                 screen_width=1920, screen_height=1080, screen_size=23.8, view_distance=60,
                 missing='nan', maxgap=75, window_size=3, smooth='median', window_length=20,
                 threshold=30, maxinterval=75, maxangle=0.5, minduration=60):

        self.rawdata = np.array([np.array(time),
                                 np.array(x),
                                 np.array(y),
                                 np.array(pupil)],
                                 dtype='float')
        self.freq = freq # sampling frequency (Hz)
        self.screen_width = screen_width # pixels
        self.screen_height = screen_height # pixels
        self.screen_size = screen_size # inch
        self.view_distance = view_distance # cm
        self.missing = missing # coding of NaN
        self.maxgap = maxgap # max gap length (ms)
        self.window_size = window_size # size of window function
        self.smooth = smooth # 'average' or 'median'
        self.window_length = window_length # size of window function (samples)
        self.threshold = threshold # velocity threshold (deg/s)
        self.maxinterval = maxinterval # max interval between fixations
        self.maxangle = maxangle # max angle between fixations
        self.minduration = minduration # minimum duration of fixations


    def fill_nan(self):

        '''
        Fill the missing value (nan)
        '''

        # detect areas of the missing data
        if self.missing == 'nan':
            miss = np.array(np.isnan(self.rawdata[1]), dtype=int)
        else:
            miss = np.array(self.rawdata[1]==missing, dtype=int)
        diff = np.diff(miss)
        starts = np.where(diff==1)[0] + 1
        ends = np.where(diff==-1)[0] + 1
        if len(starts) < len(ends):
            ends = np.delete(ends, 0)
        elif len(starts) > len(ends):
            ends = np.insert(ends, -1, len(self.rawdata[1])-1)
        else:
            pass

        # convert millisecond to frequency
        maxgap = (self.maxgap / 1000) * self.freq

        # detect nonblink data
        nonblinks = np.array([starts[(ends-starts < maxgap)]-1,
                              ends[(ends-starts < maxgap)]]).T

        # fill the missing data, except for blinks
        # with linear interpolation
        x_linspace = np.array([*map(lambda x: np.linspace(self.rawdata[1,x[0]],self.rawdata[1,x[1]],x[1]-x[0]), nonblinks)])
        y_linspace = np.array([*map(lambda x: np.linspace(self.rawdata[2,x[0]],self.rawdata[2,x[1]],x[1]-x[0]), nonblinks)])
        pupil_linspace = np.array([*map(lambda x: np.linspace(self.rawdata[3,x[0]],self.rawdata[3,x[1]],x[1]-x[0]), nonblinks)])

        self.filled = self.rawdata

        for i in range(len(nonblinks)):
            self.filled[1, nonblinks[i,0]:nonblinks[i,1]] = x_linspace[i]
            self.filled[2, nonblinks[i,0]:nonblinks[i,1]] = y_linspace[i]
            self.filled[3, nonblinks[i,0]:nonblinks[i,1]] = pupil_linspace[i]


    def smooth_data(self):

        '''
        Smooth data: Moving median or average
        '''

        self.filled = pd.DataFrame(self.filled).T
        self.filled.columns = ['time','x','y','pupil']

        # noise reduction (moving median or average)
        if self.smooth == 'median':
            self.smoothed = self.filled.rolling(window=self.window_size, min_periods=1).median()
        elif self.smooth == 'average':
            self.smoothed = self.filled.rolling(window=self.window_size, min_periods=1).mean()
        else:
            raise ValueError("This smoothing method ({}) is not supported: 'smooth' accepts 'median' or 'average.'".format(self.smooth))

        # modify timestamp
        self.smoothed.loc[:,'time'] = self.filled.loc[:,'time']


    def pix2deg(self):

        '''
        Convert pixel to degree
        '''

        # calculate the pixel size in mm
        self.pix_size = self.window_size * 25.4 / (self.screen_width**2 + self.screen_height**2)**0.5

        # calculate one visual angle in pixel
        self.deg_size = self.view_distance * 10 * math.tan(math.radians(1)) / self.pix_size


    def calculate_velocity(self):

        '''
        Calculate the angular velocity between samples
        '''

        # convert the window length from mm to samples
        window_length = round(self.freq / self.window_length)

        diff_x = self.filled['x'] - self.filled['x'].shift(window_length)
        diff_y = self.filled['y'] - self.filled['y'].shift(window_length)
        self.smoothed['ang_velocity'] = (diff_x**2 + diff_y**2)**0.5 / self.deg_size * self.freq
        self.smoothed.loc[self.smoothed['ang_velocity'] < self.threshold, 'gaze'] = 'fixation'
        self.smoothed.loc[self.smoothed['ang_velocity'] >= self.threshold, 'gaze'] = 'saccade'
        self.smoothed.loc[np.isnan(self.smoothed['ang_velocity']), 'gaze'] = 'blink'


    def merge_fixations(self):

        '''
        Merge adjacent fixations
        '''

        saccades = np.array(self.smoothed['gaze']=='saccade', dtype=int)
        diff = np.diff(saccades)
        starts = np.where(diff==1)[0] + 1
        ends = np.where(diff==-1)[0] + 1
        if len(starts) < len(ends):
            ends = np.delete(ends, 0)
        elif len(starts) > len(ends):
            ends = np.insert(ends, -1, len(self.smoothed)-1)
        else:
            pass

        # convert the max interval from mm to samples
        maxinterval = (self.maxinterval / 1000) * self.freq

        # detect too short saccades
        short_saccades = np.array([starts[(ends-starts < maxinterval)],
                                   ends[(ends-starts < maxinterval)]]).T

        # calculate angular distance between fixations
        diff_x = np.array(self.smoothed['x'][short_saccades[:,1]]) - \
            np.array(self.smoothed['x'][short_saccades[:,0]])
        diff_y = np.array(self.smoothed['y'][short_saccades[:,1]]) - \
            np.array(self.smoothed['y'][short_saccades[:,0]])
        ang_distance = np.array((diff_x**2 + diff_y**2)**0.5 / self.deg_size < self.maxangle)

        short_saccades = short_saccades[ang_distance,:]
        self.classified = copy.copy(self.smoothed)

        for i in range(len(short_saccades)):
            self.classified.loc[short_saccades[i,0]:short_saccades[i,1], 'gaze'] = 'fixation'


    def discard_fixations(self):

        '''
        Discard short fixations
        '''

        fixations = np.array(self.classified['gaze']=='fixation', dtype=int)
        diff = np.diff(fixations)
        starts = np.where(diff==1)[0] + 1
        ends = np.where(diff==-1)[0] + 1
        if len(starts) < len(ends):
            ends = np.delete(ends, 0)
        elif len(starts) > len(ends):
            ends = np.insert(ends, -1, len(self.classified)-1)
        else:
            pass

        # convert min duration from ms to samples
        minduration = (self.minduration / 1000) * self.freq

        # detect short fixations
        short_fixations = np.array([starts[(ends-starts < minduration)],
                                    ends[(ends-starts < minduration)]]).T

        for i in range(len(short_fixations)):
            self.classified.loc[short_fixations[i,0]:short_fixations[i,1], 'gaze'] = 'saccade'

    # classify gaze data into fixation, saccade, or blink
    def classify_gaze(self):
        self.fill_nan()
        self.smooth_data()
        self.pix2deg()
        self.calculate_velocity()
        self.merge_fixations()
        self.discard_fixations()
        return self.classified

    def analyze_aoi(self, aoi):

        self.aoi = None
        self.gazedata = copy.copy(self.classified)

        if type(aoi) == dict:
            aoi = pd.DataFrame(aoi).T
            aoi = aoi.reset_index()
            aoi.columns = ['aoi_name','top','right','bottom','left']
            self.aoi = aoi
        else:
            raise TypeError("'aoi' must be dictionary type.")

        if all(self.aoi['right'] > self.aoi['left']) and all(self.aoi['top'] > self.aoi['bottom']):
            pass
        else:
            raise ValueError("Specify the points of AOI correctly: 'top', 'right', 'bottom', and 'left.'")

        for i in self.aoi.index:
            self.gazedata[self.aoi.loc[i,'aoi_name']] = \
                np.array(((self.gazedata.x > self.aoi.loc[i,'left']) & (self.gazedata.x < self.aoi.loc[i,'right'])) &
                         ((self.gazedata.y > self.aoi.loc[i,'bottom']) & (self.gazedata.y < self.aoi.loc[i,'top'])) &
                         (self.gazedata.gaze == 'fixation'))
            self.gazedata[self.aoi.loc[i,'aoi_name']] = \
                self.gazedata[self.aoi.loc[i,'aoi_name']].map({True: 'in', False: 'out'})

        return self.gazedata
