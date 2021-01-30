# -*- coding: utf-8 -*-
# @Time    : 1/16/2021 8:33 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : commons.py
# @Software: src

import argparse
import cv2 as cv
import numpy as np


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


def get_program_instructions():
    return "\n ----------------------------------------------------------------------------------" \
           "\n This program implements a user interface for annotating keypoints in TSA Dataset\n" \
           "\n The program has the following 3 launch (-m) modes: \n" \
           "\tall     : launch program to load all scans\n" \
           "\tpreview : launch program to load only completely annotated scans\n" \
           "\tannotate: (Default) launch program to load only scans yet to be completed\n" \
           "\tsample  : launch program and load scans starting with a particular scan\n" \
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
