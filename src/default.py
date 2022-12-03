# -*- coding: utf-8 -*-
# @Time    : 1/16/2021 7:51 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : default.py
# @Software: src

import sys
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d # <--- This is important for 3d plotting


def valid_kypts_per_16frames():
    # Keypts:  1   2   3   4   5   6   7   8   9  10  11  12  13  14    # Frames
    #          Nk RSh REb RWr LSh LEb LWr RHp RKe RAk LHp LKe LAk MHp
    config = [[1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 0
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 1
              [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],  # 2*
              [0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],  # 3**
              [0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],  # 4
              [0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],  # 5**
              [1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],  # 6*
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 7
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 8
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 9
              [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0],  # 10*
              [0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0],  # 11**
              [0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0],  # 12
              [0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0],  # 13**
              [1,  1,  1,  1,  0,  0,  0,  1,  1,  1,  1,  1,  1,  0],  # 14*
              [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]]  # 15
    A = np.array(config, dtype=np.bool)
    # transform shape=(16, 13) to shape=(13, 16)
    return np.transpose(A) # equivalent to: np.rot90(np.fliplr(A))

def valid_kypts_per_64frames():
    config = np.ones(shape=(64, 14), dtype=np.int32)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  1] = 0  # RSh (kpt:2)
    config[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  2] = 0  # REb (kpt:3)
    config[[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  3] = 0  # RWr (kpt:4)
    config[[44, 45, 46, 47, 48, 49, 50, 51],  4] = 0  # LSh (kpt:5)
    config[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  5] = 0  # LEb (kpt:6)
    config[[43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  6] = 0  # LWr (kpt:7)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  7] = 0  # RHp (kpt:8)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  8] = 0  # RKe (kpt:9)
    config[[13, 14, 15, 16, 17, 18, 19, 20],  9] = 0  # RAk (kpt:10)
    config[[44, 45, 46, 47, 48, 49, 50, 51], 10] = 0  # LHp (kpt:11)
    config[[44, 45, 46, 47, 48, 49, 50, 51], 11] = 0  # LKe (kpt:12)
    config[[44, 45, 46, 47, 48, 49, 50, 51], 12] = 0  # LAk (kpt:13)
    config[[13, 14, 15, 16, 17, 18, 19, 20, 44, 45, 46, 47, 48, 49, 50, 51], 13] = 0  # MHp (kpt:14)
    A = np.array(config, dtype=np.bool)
    # transform shape=(64, 13) to shape=(13, 64)
    return np.transpose(A)

def set_colors(colorKeys, step=20, pad=10):
    '''
    Set a unique, distinct color (BGR) for each color key and return dictionary of key-to-color map
        Note: infinite loop may occur in current implementation
    :param colorKeys:   list of color keys
    :return:            dictionary of key-to-color map
    '''
    from random import seed
    from random import randint
    colorKeys.sort()
    seed(len(colorKeys))
    keyToColorMap = {}
    usedColors = [(0,0,0), (255,255,255)] # Black cannot be used
    b, g, r = usedColors[0]
    for i, key in enumerate(colorKeys):
        j = i
        while (b, g, r) in usedColors:
            s = (step * j) % 220
            b, g, r = randint(s, 256), randint(s, 256), randint(s, 256)
            j += 1
        keyToColorMap[key] = (b, g, r)
        for b_idx in range(-pad, pad + 1):
            for g_idx in range(-pad, pad + 1):
                for r_idx in range(-pad, pad + 1):
                    usedColors.append((b+b_idx, g+g_idx, r+r_idx))
    return keyToColorMap

def get_valid_kypts_per_frame(tsaFormat):
    if tsaFormat == 'aps':
        return valid_kypts_per_16frames()
    elif tsaFormat == 'a3daps':
        return valid_kypts_per_64frames()
    else:
        print("ERROR: Unrecognized TSA file extention: {}".format(tsaFormat))
        sys.exit(0)

def switch_dict_keys_and_values(orgDict):
    newDict = dict()
    for key, value in orgDict.items():
        newDict[value] = key
    return newDict

def kpt_ordered_annotation_stack(kptLimbOrder):
    kptStack = list(kptLimbOrder.keys())
    kptStack.reverse()
    return kptStack

def keypoints_alphabet_label(kptLimbOrder, alphabets):
    kptAlphas = dict()
    for idx, kpt in enumerate(kptLimbOrder):
        kptAlphas[kpt] = alphabets[idx]
    return kptAlphas

def convert_ord_key_identifier(alphabets):
    ordAlphaKeys = list()
    for letter in alphabets:
        ordAlphaKeys.append(ord(letter))
    return ordAlphaKeys


ALPHABETS_ID = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

KEYPOINTS_ID = ['Nk' , 'RSh', 'REb', 'RWr', 'RHp', 'RKe', 'RAk',
                'MHp', 'LSh', 'LEb', 'LWr', 'LHp', 'LKe', 'LAk']

KPTS_ID_INDX = {'Nk' :0,  'RSh':1, 'REb':2, 'RWr':3, 'RHp':7,  'RKe':8,  'RAk':9,
                'MHp':13, 'LSh':4, 'LEb':5, 'LWr':6, 'LHp':10, 'LKe':11, 'LAk':12}

LIMB_KPTS_PAIRS = {"RShoulder":('Nk','RSh'), "RBicep":('RSh','REb'), "RForearm":('REb','RWr'),
                   "LShoulder":('Nk','LSh'), "LBicep":('LSh','LEb'), "LForearm":('LEb','LWr'),
                   "RAbdomen":('RSh','RHp'), "RThigh":('RHp','RKe'), "RLeg":('RKe','RAk'),
                   "LAbdomen":('LSh','LHp'), "LThigh":('LHp','LKe'), "LLeg":('LKe','LAk'),
                   "RWaist":('MHp','RHp'), "LWaist":('MHp','LHp'), "Spine":('MHp','Nk')}

KPT_LIMB_ORDER = {'RWr':(None,None), 'REb':('RForearm',None), 'RSh':('RBicep',None),
                  'RHp':('RAbdomen',None), 'RKe':('RThigh',None), 'RAk':('RLeg',None),
                  'MHp':('RWaist',None), 'Nk':('RShoulder','Spine'),
                  'LWr':(None,None), 'LEb':('LForearm',None), 'LSh':('LBicep','LShoulder'),
                  'LHp':('LAbdomen','LWaist'), 'LKe':('LThigh',None), 'LAk':('LLeg',None)}

DEFAULT_3DPOSE = {'Nk' :[256, 200, 256], 'MHp':[256, 400, 256],
                  'RSh':[156, 200, 256], 'REb':[ 56, 150, 256], 'RWr':[156, 100, 256],
                  'LSh':[356, 200, 256], 'LEb':[456, 150, 256], 'LWr':[356, 100, 256],
                  'RHp':[200, 400, 256], 'RKe':[192, 500, 256], 'RAk':[182, 650, 256],
                  'LHp':[312, 400, 256], 'LKe':[320, 500, 256], 'LAk':[330, 650, 256]}

LS_ALGO_STATUS = {-1: 'improper input parameters status returned from MINPACK.',
                  0 : 'the maximum number of function evaluations is exceeded.',
                  1 : 'gtol termination condition is satisfied.',
                  2 : 'ftol termination condition is satisfied.',
                  3 : 'xtol termination condition is satisfied.',
                  4 : 'Both ftol and xtol termination conditions are satisfied.'}

DESCRIPTIONS = {'Nk' :'Center of neck, in-between the head and shoulder blade',
                'RSh':'Epicenter of the right shoulder joint',
                'REb':'Epicenter of the right elbow joint',
                'RWr':'Epicenter of the right wrist joint, just beneath the palm',
                'RHp':'The right hip joint, along the waist line',
                'RKe':'Epicenter of the right knee joint',
                'RAk':'The right ankle joint, just above the ball of the feet',
                'MHp':'Mid-hip point, intersection of the spine and waist line',
                'LSh':'Epicenter of the left shoulder joint',
                'LEb':'Epicenter of the left elbow joint',
                'LWr':'Epicenter of the left wrist joint, just beneath the palm',
                'LHp':'The left hip joint, along the waist line',
                'LKe':'Epicenter of the left knee joint',
                'LAk':'The left ankle joint, just above the ball of the feet'}

PAY_RATE = .05 # $ per scan
EMPTY_CELL = ''
SUBSETS = ['train']
TSA_FORMAT = 'aps'
RUN_MODES = {'all': ['complete', 'incomplete', 'faulty', EMPTY_CELL],
             'annotate': ['incomplete', 'faulty', EMPTY_CELL],
             'sample': ['incomplete', 'faulty', EMPTY_CELL],
             'preview': ['complete'],
             'faulty': ['faulty']}

FRM_HGT = 660
FRM_WDT = 512
K_DEFAULT = (-1, -1, 0)
BRIGT_INCR = 5
BRCNT_INCR = 4
SCALE_INCR = 0.1
CONTR_INCR = 0.5
NUM_OF_FRAMES = 16
AUTO_SAVE_FREQ = 180 # in secs
WINDOW_SHAPE = (660, 1200, 3)
WINDOW_TOP_LEFT = (550, 150)
N_KEYPOINTS = len(KEYPOINTS_ID)

PENDING_COLOR = (  0, 200, 200) # yellow NEXTKPT_COLOR
FAULTYK_COLOR = ( 50,   0, 200) # red    PENDING_COLOR
INVALID_COLOR = (235, 235, 235) # paper-white
MARKEDK_COLOR = (100, 200,   0) # green
KSHADOW_COLOR = (  0,   0,   0) # black
LFTSIDE_COLOR = (128,   0, 128) # purple
RGTSIDE_COLOR = (  0, 100, 128) # olive
MIDSIDE_COLOR = ( 32,  32,  32) # space-black
NEXTKPT_COLOR = ( 70,   3,  54) # tsa-aps purple [87,4,68]
WCANVAS_COLOR = 235 # paper-white

# global ax subplot
fig = plt.figure()
AX_SP = plt.axes(projection='3d')
POSE_FIG_HGT = 480 # actual hgt is 440
POSE_FIG_WDT = 480 # actual wdt is 640
POSE_WIN_TLY = 50  # top-left position x component
POSE_WIN_TLX = 710 # top-left position y component

FRAME_VALID_KPTS = get_valid_kypts_per_frame(TSA_FORMAT)
KPTS_INDX_ID = switch_dict_keys_and_values(KPTS_ID_INDX)
KPTS_STACK = kpt_ordered_annotation_stack(KPT_LIMB_ORDER)
KPTS_ALPHA = keypoints_alphabet_label(KPT_LIMB_ORDER, ALPHABETS_ID)
ALPHA_KPTS = switch_dict_keys_and_values(KPTS_ALPHA)
ALPHABET_ORD = convert_ord_key_identifier(ALPHABETS_ID)
KPT_COLORS = set_colors(KEYPOINTS_ID, step=17)


# REPLACE 'John' with your first name
MARKER_NAME = 'John'
START_DF_IDX = 0
STOP_DF_IDX = 99
MAX_AGG_ERROR = 200 # max allowed aggregate(sum) error per keypoint
MAX_KPT_ERROR = 20  # max allowed error per keypoint annotation
INFO_TEXT = '{nScans:>4} Completed Scans  Earnings: ${payCost:<6.2f}  ' \
            'Avg. Error: {avgError:<4.1f}  Total Time: {time:.1f} hrs'
npyHgt, npyWdt = 165, 128 # 330, 256

IMAGES_ROOT_DIR = '../../datasets/tsa/aps_images/dataset/{}_set'.format(SUBSETS[0])
IMAGES_NPFILE = '../data/tsa/{}/{}Set_{}x{}.npy'.format(TSA_FORMAT, SUBSETS[0], npyHgt, npyWdt)
KPTS_CSV_FILE = '../data/csvs/{}KeypointsAnnotations{}.csv'.format(TSA_FORMAT, '{}')
TEMP_CSV_FILE = '../data/csvs/{}Template.csv'.format(TSA_FORMAT)
METADATA_PATH = '../data/meta/meta{}.pickle'
POSE3D_FIG_PATH = '../data/meta/pose3d.png'
