# -*- coding: utf-8 -*-
# @Time    : 1/16/2021 8:33 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : commons.py
# @Software: src

import os
import sys
import argparse
import cv2 as cv
import numpy as np
import pandas as pd

sys.path.append('../')
from default import TSA_FORMAT, TEMP_CSV_FILE, NUM_OF_FRAMES, FRM_HGT, FRM_WDT


def create_display_window(name, x_pos, y_pos, x_size=384, y_size=495):
    cv.namedWindow(name, cv.WINDOW_NORMAL + cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow(name, x_size, y_size)
    cv.moveWindow(name, x_pos, y_pos)


def define_windows(winNames, winPerRow, rWdt, rHgt, xStart=0, yStart=0, titleHgt=35):
    #assert (isinstance(winPerRow, int))
    winNames.sort()
    cv.destroyAllWindows()
    for i in range(len(winNames)):
        x_p = xStart + (i % winPerRow) * rWdt
        y_p = yStart + int(i / winPerRow) * (rHgt + titleHgt) #345
        create_display_window(winNames[i], x_p, y_p, x_size=rWdt, y_size=rHgt)

def get_png_frame_image(scanRootDir, scanId, fid):
    imgPath = os.path.join(scanRootDir, scanId, '{}.png'.format(fid))
    try:
        frmImg = cv.imread(imgPath)
        assert (np.max(frmImg) > np.min(frmImg))
        return frmImg
    except: #IOError
        print('\tImage Error: filepath may not exist: {}'.format(imgPath))
        return None

def get_npy_frame_image(npyFile, scanNpIdx, fid):
    try:
        assert(not pd.isna(scanNpIdx))
        scanNpIdx = int(scanNpIdx)
        frmImg = npyFile[scanNpIdx, fid]
        frmImg = cv.resize(frmImg, (FRM_WDT, FRM_HGT),
                           interpolation=cv.INTER_CUBIC)  # cv.INTER_AREA
        return frmImg
    except: #IOError
        print('\tScan does not have a valid numpy index: {}'.format(scanNpIdx))
        return None


def get_program_instructions():
    return "\n ----------------------------------------------------------------------------------" \
           "\n This program implements a user interface for annotating keypoints in TSA Dataset\n" \
           "\n The program has the following 5 launch (-m) modes: \n" \
           "\tall     : launch program to load all scans\n" \
           "\tpreview : launch program to load only completely annotated scans\n" \
           "\tannotate: (Default) launch program to load only scans yet to be completed\n" \
           "\tsample  : launch and load scans to annotate starting with the arg passed scanID\n" \
           "\tfaulty  : launch program loading all faulty scans with unacceptable annotations\n" \
           "\n Below are the program's mouse controls\n" \
           "\tleft-click : mark keypoint with mouse by hovering and left-clicking on it\n" \
           "\tright-click: right-clicking on the mouse will move to next frame, or scan if done\n" \
           "\tmousewheel : zoom in/out on the image by rolling your mousewheel\n" \
           "\n Below are the program's key controls\n" \
           "\tA-to-N       : undo marking of keypoint corresponding to the letter\n" \
           "\tu/backspace  : undo most recent keypoint marking\n" \
           "\tr/delete     : delete all keypoint markings in the frame\n" \
           "\tn/right-arrow: skip to next frame image of scan\n" \
           "\tp/left-arrow : return to previously visited frame image\n" \
           "\tf/enter      : force advance to the next scan when annotation of scan is complete\n" \
           "\th/up-arrow   : enhance image contrast\n" \
           "\tl/down-arrow : reduce image contrast\n" \
           "\tb            : brighten image\n" \
           "\td            : darken image\n" \
           "\te/esc        : forcefully exit the program\n" \
           "\ti/space-bar  : display program help information\n" \
           " ----------------------------------------------------------------------------------\n"


def runtime_args():
    '''
    runtime arguments for train_nn.py in training
    :return: runtime arguments dictionary
    '''
    ap = argparse.ArgumentParser(description='Train zone threat detection network')

    # general
    ap.add_argument("-m", "--sampleMode", type=str, default='annotate',
                    help="set scans to load for annotation or preview")
    ap.add_argument("-s", "--scanSample", type=str, default='',
                    help="scan sample to load first, must be set if '-m'='sample'")

    # args = vars(ap.parse_args()) # Converts to dictionary format
    args = ap.parse_args()
    return args


def encode_dataset():
    # read and encode dataset
    subset = 'train'
    dsHgt, dsWdt = 165, 128 # 110, 85
    scanRootDir = '../../datasets/tsa/aps_images/dataset/{}_set'.format(subset)
    wrtPath = '../data/tsa/{}/{}Set_{}x{}.npy'.format(TSA_FORMAT, subset, dsHgt, dsWdt)
    dfKpt = pd.read_csv(TEMP_CSV_FILE)
    scanIdList = os.listdir(scanRootDir)
    nScans = len(scanIdList)
    imgData = np.zeros(shape=(nScans,NUM_OF_FRAMES,dsHgt,dsWdt,3), dtype=np.uint8)

    # encode images
    nImages = 0
    for idx, scanId in enumerate(scanIdList):
        for fid in range(NUM_OF_FRAMES):
            imgPath = os.path.join(scanRootDir, scanId, '{}.png'.format(fid))
            try:
                frmImg = cv.imread(imgPath)
                assert (np.max(frmImg) > np.min(frmImg))
                frmImg = cv.resize(frmImg, (dsWdt, dsHgt), interpolation=cv.INTER_CUBIC)
                imgData[idx, fid] = frmImg
                dfKpt.loc[dfKpt['scanID']==scanId, 'Subset'] = subset
                dfKpt.loc[dfKpt['scanID']==scanId, 'npIndex'] = idx
                nImages += 1
            except: #IOError
                print('\tImage Error: filepath may not exist: {}'.format(imgPath))

        if (idx+1)%100==0 or (idx+1)==nScans:
            print('{:>7} of {} scans encoded..'.format(idx+1, nScans))

    # save csv and numpy file of encoded images
    np.save(wrtPath, imgData)
    dfKpt.to_csv(TEMP_CSV_FILE, encoding='utf-8', index=False)



if __name__ == "__main__":
    encode_dataset()