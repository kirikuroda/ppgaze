import copy
import math
import numpy as np
import pandas as pd

class GazeData:
    
    def __init__(self, time, x, y, pupil, freq,
                 width=1920, height=1080, size=23.8, distance=60, 
                 missing='nan', maxgap=75, win_smooth=3, smooth='median'):
        
        self.rawdata = np.array([np.array(time),
                                 np.array(x),
                                 np.array(y),
                                 np.array(pupil)],
                                 dtype='float')
        self.freq = freq # sampling frequency (Hz)
        self.width = width # monitor width (pix)
        self.height = height # monitor height (pix)
        self.size = size # monitor size (inch)
        self.distance = distance # viewing distance (cm)
        self.missing = missing # coding of NaN
        self.maxgap = maxgap # max gap length (ms)
        self.win_smooth = win_smooth # size of window function
        self.smooth = smooth # 'average' or 'median'
        
    
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
            if ends[0] < starts[0]:
                starts = np.delete(starts, -1)
                ends = np.delete(ends, 0)
        
        # convert millisecond to samples
        maxgap = (self.maxgap / 1000) * self.freq
        
        # detect nonblink data
        nonblinks = np.array([starts[(ends-starts < maxgap)],
                              ends[(ends-starts < maxgap)]]).T
        
        # fill the missing data, except for blinks
        # with linear interpolation
        x_linspace = \
            np.array([*map(lambda x: np.linspace(self.rawdata[1,x[0]-1],self.rawdata[1,x[1]],x[1]-x[0]+2), nonblinks)])
        y_linspace = \
            np.array([*map(lambda x: np.linspace(self.rawdata[2,x[0]-1],self.rawdata[2,x[1]],x[1]-x[0]+2), nonblinks)])
        pupil_linspace = \
            np.array([*map(lambda x: np.linspace(self.rawdata[3,x[0]-1],self.rawdata[3,x[1]],x[1]-x[0]+2), nonblinks)])
        
        self.filled = self.rawdata
        
        nonblinks = np.array([*map(lambda x: range(x[0]-1,x[1]+1), nonblinks)])
        
        dim = 0
        if len(nonblinks) > 0:
            for i in nonblinks:
                self.filled[1, i] = x_linspace[dim]
                self.filled[2, i] = y_linspace[dim]
                self.filled[3, i] = pupil_linspace[dim]
                dim += 1
        else:
            pass
        
    
    def smooth_data(self):
        
        '''
        Smooth data: Moving median or average
        '''
        
        self.filled = pd.DataFrame(self.filled).T
        self.filled.columns = ['time','x','y','pupil']
        
        # noise reduction (moving median or average)
        if self.smooth == 'median':
            self.smoothed = self.filled.rolling(window=self.win_smooth).median()
        elif self.smooth == 'average':
            self.smoothed = self.filled.rolling(window=self.win_smooth).mean()
        else:
            raise ValueError("This smoothing method ({}) is not supported: 'smooth' accepts 'median' or 'average.'".format(self.smooth))
        
        # modify timestamp
        self.smoothed['time'] = self.filled['time']
        
    
    def pix2deg(self):
        
        '''
        Convert pixel to degree
        '''
        
        # calculate the pixel size in mm
        self.pix_size = self.size * 25.4 / (self.width**2 + self.height**2)**0.5
        
        # calculate one visual angle in pixel
        self.deg_size = self.distance * 10 * math.tan(math.radians(1)) / self.pix_size
    
    
    def calculate_velocity(self):
        
        '''
        Velocity-based filter
        Calculate the angular velocity between samples
        '''
        
        # convert the window length from mm to samples
        win_velocity = round(self.win_velocity / 1000 * self.freq)
        
        diff_x = \
            np.array(self.smoothed['x'] - self.smoothed['x'].shift(win_velocity))
        diff_y = \
            np.array(self.smoothed['y'] - self.smoothed['y'].shift(win_velocity))
        ang_velocity = (((diff_x**2 + diff_y**2)**0.5) / self.deg_size) * (1000 / self.win_velocity)
        ang_velocity[np.isnan(ang_velocity)] = 0
        self.smoothed['ang_velocity'] = ang_velocity
        self.smoothed['gaze'] = 'blink'
        self.smoothed.loc[(self.smoothed['ang_velocity'] > 0) & \
                          (self.smoothed['ang_velocity'] < self.threshold_velocity), 'gaze'] = 'fixation'
        self.smoothed.loc[self.smoothed['ang_velocity'] >= self.threshold_velocity, 'gaze'] = 'saccade'
        self.smoothed.loc[self.smoothed['ang_velocity'] == 0, ['x','y','pupil']] = np.nan
        
    
    def calculate_dispersion(self):
        
        '''
        Dispersion-based filter
        '''
        
        gaze_pos = np.array([np.array(self.smoothed['x']), 
                             np.array(self.smoothed['y'])])
        
        min_win_dispersion = round(self.win_dispersion / 1000 * self.freq)
        win_dispersion = round(self.win_dispersion / 1000 * self.freq)
        start = 0
        gaze = ['blink'] * len(self.smoothed)
        data_length = len(self.smoothed) - 1
        while start + win_dispersion <= data_length:
            if np.isnan(gaze_pos[0,start]):
                start += 1
            else:
                end = start + win_dispersion
                diff_x = np.max(gaze_pos[0,start:end])-np.min(gaze_pos[0,start:end])
                diff_y = np.max(gaze_pos[1,start:end])-np.min(gaze_pos[1,start:end])
                dispersion = (diff_x + diff_y) / self.deg_size
                
                if dispersion < self.threshold_dispersion:
                    if end == data_length:
                        gaze[start:end+1] = ['fixation']*(end-start+1)
                        x = [np.median(gaze_pos[0,start:end+1])]*(end-start*1)
                        y = [np.median(gaze_pos[1,start:end+1])]*(end-start*1)
                        gaze_pos[0,start:end] = x
                        gaze_pos[1,start:end] = y
                    win_dispersion += 1
                else:
                    if win_dispersion > min_win_dispersion:
                        gaze[start:end-1] = ['fixation']*(end-start-1)
                        x = [np.mean(gaze_pos[0,start:end-1])]*(end-start-1)
                        y = [np.mean(gaze_pos[1,start:end-1])]*(end-start-1)
                        gaze_pos[0,start:end-1] = x
                        gaze_pos[1,start:end-1] = y
                        win_dispersion = min_win_dispersion
                        start = end - 1
                    else:
                        if end == data_length:
                            gaze[start:end+1] = ['saccade']*(end-start+1)
                        gaze[start] = 'saccade'
                        start += 1
        self.smoothed['x_fixation'] = gaze_pos[0,:]
        self.smoothed['y_fixation'] = gaze_pos[1,:]
        self.smoothed['gaze'] = gaze
        
    
    def merge_fixations(self):
        
        '''
        Merge adjacent fixations
        '''
        
        nonfixations = np.array(self.smoothed['gaze']!='fixation', dtype=int)
        
        diff = np.diff(nonfixations)
        starts = np.where(diff==1)[0] + 1
        ends = np.where(diff==-1)[0] + 1
        if len(starts) < len(ends):
            ends = np.delete(ends, 0)
        elif len(starts) > len(ends):
            ends = np.insert(ends, -1, len(self.smoothed)-1)
        else:
            if ends[0] < starts[0]:
                starts = np.delete(starts, -1)
                ends = np.delete(ends, 0)
        
        # convert the max interval from mm to samples
        maxinterval = (self.maxinterval / 1000) * self.freq
        
        # detect too short saccades
        short_nonfixations = np.array([starts[(ends-starts < maxinterval)],
                                       ends[(ends-starts < maxinterval)]]).T
        
        # calculate angular distance between fixations
        diff_x = \
            np.array(self.smoothed['x'][short_nonfixations[:,1]]) - \
            np.array(self.smoothed['x'][short_nonfixations[:,0]-1])
        diff_y = \
            np.array(self.smoothed['y'][short_nonfixations[:,1]]) - \
            np.array(self.smoothed['y'][short_nonfixations[:,0]-1])
        ang_distance = np.array((diff_x**2 + diff_y**2)**0.5 / self.deg_size < self.maxangle)
        short_nonfixations = short_nonfixations[np.where(ang_distance==True)[0]]
        
        self.classified = copy.copy(self.smoothed)
        
        gaze = np.array(self.classified['gaze'])
        if len(short_nonfixations) > 0:
            short_nonfixations = np.array([*map(lambda x: range(x[0],x[1]), short_nonfixations)])
            for i in range(len(short_nonfixations)):
                gaze[short_nonfixations[i]] = 'fixation'
            self.classified['gaze'] = gaze
        else:
            pass
        
    
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
            ends = np.append(ends, len(self.classified)-1)
        else:
            if ends[0] < starts[0]:
                starts = np.delete(starts, -1)
                ends = np.delete(ends, 0)
        
        # convert min duration from ms to samples
        minduration = (self.minduration / 1000) * self.freq
        
        # detect short fixations
        short_fixations = np.array([starts[(ends-starts < minduration)],
                                    ends[(ends-starts < minduration)]]).T
        
        short_fixations = np.array([*map(lambda x: range(x[0],x[1]), short_fixations)])
        
        gaze = np.array(self.classified['gaze'])
        
        if len(short_fixations) > 0:
            for i in short_fixations:
                gaze[i] = 'saccade'
            self.classified['gaze'] = gaze
        
    
    
    def velocity_filter(self, threshold_velocity=30, win_velocity=20, maxinterval=75, 
                        maxangle=0.5, minduration=60):
        self.threshold_velocity = threshold_velocity
        self.win_velocity = win_velocity
        self.maxinterval = maxinterval
        self.maxangle = maxangle
        self.minduration = minduration
        self.fill_nan()
        self.smooth_data()
        self.pix2deg()
        self.calculate_velocity()
        self.merge_fixations()
        self.discard_fixations()
        return self.classified
    
    def dispersion_filter(self, threshold_dispersion = 1.0, win_dispersion=100, 
                          maxinterval=75, maxangle=0.5, minduration=60):
        self.threshold_dispersion = threshold_dispersion
        self.win_dispersion = win_dispersion
        self.maxinterval = maxinterval
        self.maxangle = maxangle
        self.minduration = minduration
        self.fill_nan()
        self.smooth_data()
        self.pix2deg()
        self.calculate_dispersion()
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


