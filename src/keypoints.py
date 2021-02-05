'''
Annotate keypoints in TSA Passenger Screening Dataset
'''

import os
import sys
import ast
import time
import pickle
import random
import threading
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
from reconstruct import project_keypoints_from_3d, plot_3d_pose
from commons import get_npy_frame_image, get_png_frame_image
from commons import define_windows, get_program_instructions, runtime_args
from default import IMAGES_NPFILE, IMAGES_ROOT_DIR, DESCRIPTIONS, RUN_MODES, K_DEFAULT
from default import SCALE_INCR, CONTR_INCR, KPT_LIMB_ORDER, LIMB_KPTS_PAIRS, KPT_COLORS
from default import KPTS_CSV_FILE, TEMP_CSV_FILE, METADATA_PATH, POSE3D_FIG_PATH, SUBSETS
from default import POSE_FIG_WDT, POSE_FIG_HGT, POSE_WIN_TLY, POSE_WIN_TLX, MAX_AGG_ERROR
from default import LFTSIDE_COLOR, RGTSIDE_COLOR, MIDSIDE_COLOR, FAULTYK_COLOR, WCANVAS_COLOR
from default import NEXTKPT_COLOR, PENDING_COLOR, INVALID_COLOR, MARKEDK_COLOR, KSHADOW_COLOR
from default import KEYPOINTS_ID, KPTS_ID_INDX, KPTS_STACK, N_KEYPOINTS, BRIGT_INCR, BRCNT_INCR
from default import KPTS_ALPHA, ALPHA_KPTS, ALPHABETS_ID, ALPHABET_ORD, START_DF_IDX, STOP_DF_IDX
from default import MARKER_NAME, EMPTY_CELL, NUM_OF_FRAMES, FRAME_VALID_KPTS, PAY_RATE, INFO_TEXT
from default import AUTO_SAVE_FREQ, WINDOW_SHAPE, WINDOW_TOP_LEFT, FRM_WDT, FRM_HGT, MAX_KPT_ERROR


def save_progress():
    # permanently writes changes from memory to disk
    global _updateSinceLastAutoSave
    # save keypoint annotations
    _dfKpt.to_csv(KPTS_CSV_FILE, encoding='utf-8', index=False)
    # save metadata information
    with open(METADATA_PATH, 'wb') as file_handle:
        pickle.dump(_markerMetadata, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    # reset auto-save tracker
    _updateSinceLastAutoSave = False # reset change

def periodic_save():
    # periodically save data frame by writing to csv and json
    if _updateSinceLastAutoSave:
        save_progress()
        print ("\tAlert: Periodic auto-save on {}".format(time.ctime()))
    # sleep for X seconds then recursively call function
    time.sleep(AUTO_SAVE_FREQ)
    periodic_save()

def visualize_mouse_event(eventPackage):
    global _mouseEventPack
    windowName, fid, eventCode = eventPackage
    _mouseEventPack[2] = 0
    if eventCode == 1:
        if len(_pendingKpts[fid]) >= 0:
            refresh_display(windowName, fid)
    elif eventCode == 2:
        if len(_pendingKpts[fid]) <= 0:
            # first try to move to next scan if conditions are met
            # ie. all frames have been marked with acceptable annotations
            next_scan(fid)
            if _stayOnScan: # true if conditions aren't satisfied
                # automatically move to next frame set
                slide(fid, direction=1)
    elif eventCode == 3:
        refresh_display(windowName, fid)

def update_scan_status(isGoodAnnotation):
    global _updateSinceLastAutoSave, _markerMetadata
    scanStatus = _dfKpt.loc[_dfKpt['scanID']==_scanId, 'Status'].values[0]
    if pd.isna(scanStatus) and scanStatus!='complete' and isGoodAnnotation:
        # this is a newly completed scan (NOT a revision) with acceptable annotations
        _markerMetadata['completedScans'] += 1
    # mark unique scanID as complete or faulty
    scanStatus = 'complete' if isGoodAnnotation else 'faulty'
    _dfKpt.loc[_dfKpt['scanID']==_scanId, 'Status'] = scanStatus
    _updateSinceLastAutoSave = True
    print ("\tscan:{} annotation is {} {} will be auto-saved in about {} minutes"
           .format(_scanId, scanStatus, 'and' if isGoodAnnotation else 'but', AUTO_SAVE_FREQ//60))

def update_dataframe_record(fid):
    global _updateSinceLastAutoSave
    # update keypoints in frame
    frmName = 'Frame{}'.format(fid)
    frmKptsDict = dict()
    # ensure only valid keypoint markings (x,y)>=0 is recorded
    for kpt, metaInfo in _dictOfKpts[fid].items():
        if metaInfo[0]>=0 and metaInfo[1]>=0:
            frmKptsDict[kpt] = metaInfo

    if len(frmKptsDict.keys())>0:
        # record in data frame
        _dfKpt.loc[_dfKpt['scanID']==_scanId, frmName] = str(frmKptsDict)
    else:
        # empty cell in data frame
        _dfKpt.loc[_dfKpt['scanID']==_scanId, frmName] = EMPTY_CELL
        # reset value because there is no markings of frame
        _dfKpt.loc[_dfKpt['scanID']==_scanId, 'Status'] = 'incomplete'
    _updateSinceLastAutoSave = True

def record_keypoint_meta(kpt, fid, x, y, e, onlyError=False):
    global _dictOfKpts, _changeInScan, _markerMetadata
    kptIdx = KPTS_ID_INDX[kpt]
    if onlyError:
        assert (isinstance(e, float)), 'keypoint annotation error:{} must be a float'.format(e)
        _markerMetadata['totalError'][kptIdx] += e - _dictOfKpts[fid][kpt][2]  # update error
        _dictOfKpts[fid][kpt][2] = e  # (x, y, e)
    else:
        assert (isinstance(x, int)), 'keypoint annotation x:{} must be an integer'.format(x)
        assert (isinstance(y, int)), 'keypoint annotation y:{} must be an integer'.format(y)
        _dictOfKpts[fid][kpt][:2] = [x, y]  # (x, y, e)
    _changeInScan = True

def undo_previous_marking(windowName, fid):
    global _dictOfKpts, _pendingKpts
    lastKpt = previously_marked_keypoint(fid)
    if lastKpt is not None:
        _dictOfKpts[fid][lastKpt][:2] = K_DEFAULT[:2]  # don't reset error coz total error tracking
        _pendingKpts[fid].append(lastKpt)  # set previous kpt to immediate pending kpt
        refresh_display(windowName, fid)

def reset_markings(windowName, fid):
    global _dictOfKpts, _pendingKpts
    # reset annotations to default values
    for kpt in _dictOfKpts[fid].keys():
        _dictOfKpts[fid][kpt][:2] = K_DEFAULT[:2]  # don't reset error coz total error tracking
    _pendingKpts[fid] = KPTS_STACK.copy()
    remove_invalid_keypoints(fid)
    refresh_display(windowName, fid)

def delete_keypoint_marking(windowName, fid, alphaId):
    global _dictOfKpts, _pendingKpts
    kpt = ALPHA_KPTS[alphaId]
    _dictOfKpts[fid][kpt][:2] = K_DEFAULT[:2]  # don't reset error because of total error tracking
    _pendingKpts[fid].append(kpt)  # set kpt to immediate pending kpt
    refresh_display(windowName, fid)

def slide(fid, direction=1):
    global _moveDirection, _pauseForEvent
    update_dataframe_record(fid)  # save marked coordinates
    _moveDirection = direction
    _pauseForEvent = False

def next_scan(fid, forceSkip=False):
    global _pauseForEvent, _stayOnScan

    goodMarkings = True
    scanComplete = True
    for frm in range(NUM_OF_FRAMES):
        if len(_pendingKpts[frm]) > 0:
            scanComplete = False
            break

    if scanComplete:
        for kpt in KEYPOINTS_ID:
            errorAgg = 0
            kptIdx = KPTS_ID_INDX[kpt]
            for fid in range(NUM_OF_FRAMES):
                if FRAME_VALID_KPTS[kptIdx, fid]:
                    kptError = _dictOfKpts[fid][kpt][2]
                    errorAgg += kptError
                    if kptError > MAX_KPT_ERROR:
                        goodMarkings = False
                        break
            if errorAgg>MAX_AGG_ERROR or not goodMarkings:
                goodMarkings = False
                break

    if forceSkip:
        moveToNextScan = scanComplete
    else: moveToNextScan = scanComplete and goodMarkings

    if moveToNextScan:
        update_dataframe_record(fid)
        log_scan_updated_keypoints_error()
        update_scan_status(goodMarkings)  # scan is Marked only if all frames are reviewed
        _pauseForEvent = False
        _stayOnScan = False

def terminate_program(fid):
    print("\n Program is shutting down...")
    update_dataframe_record(fid)  # save marked coordinates
    log_scan_updated_keypoints_error()
    print("\tscan will not be marked as 'completed'\n\tsaving progress...")
    save_progress()
    print("\tsession's progress has been saved successfully saved\n\tTerminating Program...")
    print("\n Thanks for putting in the time. Till next session, goodbye {}.".format(MARKER_NAME))
    cv.destroyAllWindows()  # close all open windows
    if os.path.exists(POSE3D_FIG_PATH): os.remove(POSE3D_FIG_PATH)
    plt.close()
    sys.exit()

def log_scan_updated_keypoints_error():
    # record kpts error alongside coordinates in each frame with changes
    for fid in range(NUM_OF_FRAMES):
        update_dataframe_record(fid)

    # record up-to-date 3d-pose
    if np.any(_3dPose >= 0):
        poseDict = dict()
        pose3dError = np.around(_aggErrorPerKpt, 2)
        for kpt, kptIdx in KPTS_ID_INDX.items():
            poseDict[kpt] = tuple(_3dPose[kptIdx]) + (pose3dError[kptIdx],)
        _dfKpt.loc[_dfKpt['scanID']==_scanId, '3dPose'] = str(poseDict)

def transform_point(xcord, ycord, transMatrix):
    # applies transformation matrix on 2D point
    hpoint = transMatrix.dot([xcord, ycord, 1])
    transX = int(hpoint[0] / hpoint[2])
    transY = int(hpoint[1] / hpoint[2])
    return transX, transY

def image_transform(copiedImg, transMatrix):
    # applies transformation matrix to entire image
    if np.array_equal(transMatrix, np.identity(3)):
        return copiedImg
    else:
        rows = copiedImg.shape[0]
        cols = copiedImg.shape[1]
        return cv.warpAffine(copiedImg, transMatrix[:2, :], (cols, rows))

def adjsut_brightness(image, brightness, contrast):
    # Lighten/darken image by adjusting image brightness and contrast
    assert(-127<=brightness<=127 and -127<=contrast<=127), '{} v. {}'.format(brightness, contrast)
    img = np.int32(image)
    img = img * (contrast / 127 + 1) - contrast + brightness
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def adjust_contrast(image, factor):
    # enhance image contrast to make more visible
    if factor==0: return image
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=factor, tileGridSize=(5,5))
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB) # convert from BGR to LAB color space
    l, a, b = cv.split(lab) # split on 3 different channels
    l2 = clahe.apply(l) # apply CLAHE to the L-channel
    lab = cv.merge((l2, a, b)) # merge channels
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR) # convert from LAB to BGR

def highlight_next_keypoint_suggested_position(frmImage, fid, nextKpt, radius=64):
    # overlay circular mask indicating suggested keypoint position
    if nextKpt is not None:
        kptIdx = KPTS_ID_INDX[nextKpt]
        mask = np.zeros(shape=(FRM_HGT, FRM_WDT, 3), dtype=np.uint8)
        suggestedLoc = _projectedKptsPerFrm[kptIdx][fid]
        if np.all(suggestedLoc >= 0):
            # kpt has been projected to frame
            radius = int(radius * _scaleFactor)
            x, y = transform_point(suggestedLoc[0], suggestedLoc[1], _transMatrix)
            cv.circle(mask, (x, y), radius, (255,255,255), thickness=-1, lineType=LINE2)
        return cv.addWeighted(frmImage, 0.8, mask, 0.2, 0)
    return frmImage

def frame_visual_update(fid, yMargin=30, xMargin=520, xSize=180, ySize=40):
    global _updatePoseImage, _poseImage
    winImg = np.full(WINDOW_SHAPE, fill_value=WCANVAS_COLOR, dtype=np.uint8)
    nPending = len(_pendingKpts[fid])
    nextKpt = None if nPending==0 else _pendingKpts[fid][nPending-1]  # at end of list (stack)

    # update frame image
    frmImage = adjust_contrast(_curFrmImage, factor=_contrastFtr)
    frmImage = image_transform(frmImage, _transMatrix)
    frmImage = adjsut_brightness(frmImage, _brightenFtr, _brigContFtr)

    # update 3d-pose image
    xStart, xEnd = POSE_WIN_TLX, POSE_WIN_TLX+POSE_FIG_WDT
    yStart, yEnd = POSE_WIN_TLY, POSE_WIN_TLY+POSE_FIG_HGT
    if os.path.exists(POSE3D_FIG_PATH):
        if _updatePoseImage:
            _updatePoseImage = False
            _poseImage = cv.imread(POSE3D_FIG_PATH)[:, 90:570]
            _poseImage = np.where(_poseImage==255, WCANVAS_COLOR, _poseImage)
        winImg[yStart:yEnd, xStart:xEnd] = _poseImage
    else: winImg[yStart:yEnd, xStart:xEnd] = WCANVAS_COLOR

    # Frame title label
    title = 'Frame: {:<2}'.format(fid)
    labPos = (xMargin, yMargin)
    cv.putText(winImg, title, labPos, FONT1, 0.8, (  0,  0,  0), 2, lineType=LINE1)
    cv.putText(winImg, title, labPos, FONT1, 0.8, (255,255,255), 1, lineType=LINE1)

    # display metadata information
    if nextKpt is not None:
        kptDescription = ['{} Description:'.format(nextKpt), DESCRIPTIONS[nextKpt]]
        for idx, info in enumerate(kptDescription):
            yPos, fSize = 500 + (30*idx), 1. - (0.1*idx)
            txtPos = (720, yPos)
            cv.putText(winImg, info, txtPos, FONT2, fSize, KPT_COLORS[nextKpt], 2, lineType=LINE1)
            cv.putText(winImg, info, txtPos, FONT2, fSize, KSHADOW_COLOR, 1, lineType=LINE1)
    infoText = INFO_TEXT.format(nScans=_markerMetadata['completedScans'],
                                payCost=_markerMetadata['completedScans']*PAY_RATE,
                                avgError=np.mean(_markerMetadata['totalError']),
                                time=_markerMetadata['totalTime']/3600)
    cv.putText(winImg, infoText, (xMargin,640), FONT1, 0.65, (0,0,0), 1, lineType=LINE1)

    # draw edges between keypoints
    frmKptsDict = _dictOfKpts[fid]
    for limb, kptPair in LIMB_KPTS_PAIRS.items():
        kptA, kptB = kptPair
        if kptA[0]=='R' or kptB[0]=='R': lineColor = RGTSIDE_COLOR
        elif kptA[0]=='L' or kptB[0]=='L': lineColor = LFTSIDE_COLOR
        else: lineColor = MIDSIDE_COLOR
        kptAInfo = frmKptsDict.get(kptA, None)
        kptBInfo = frmKptsDict.get(kptB, None)
        if kptAInfo is not None and kptBInfo is not None:
            xKptA, yKptA = kptAInfo[:2]
            xKptB, yKptB = kptBInfo[:2]
            if xKptA>=0 and yKptA>=0 and xKptB>=0 and yKptB>=0:
                xA, yA = transform_point(xKptA, yKptA, _transMatrix)
                xB, yB = transform_point(xKptB, yKptB, _transMatrix)
                cv.line(frmImage, (xA, yA), (xB, yB), (0, 0, 0), 4)
                cv.line(frmImage, (xA, yA), (xB, yB), lineColor, 2)

    # Keypoint labels
    ym = 20
    for idx, kpt in enumerate(KPT_LIMB_ORDER.keys()):
        kptIdx = KPTS_ID_INDX[kpt]
        kptAlpha = KPTS_ALPHA[kpt]
        kptAggError = _aggErrorPerKpt[kptIdx]
        kptFrmError = 0

        # keypoint location
        if frmKptsDict.get(kpt, None) is not None:
            xKpt, yKpt, kptFrmError = frmKptsDict[kpt]
            # pinpoint keypoint on image
            if xKpt>=0 and yKpt>=0:
                tx, ty = transform_point(xKpt, yKpt, _transMatrix)
                cv.circle(frmImage, (tx, ty), 7, KSHADOW_COLOR, thickness=-1, lineType=LINE2)
                cv.circle(frmImage, (tx, ty), 5, KPT_COLORS[kpt], thickness=-1, lineType=LINE2)

        # keypoint information box
        yPosA = yMargin + ((idx + 1) * ySize)
        yPosB = yMargin + ((idx + 2) * ySize)
        kptRecA = (xMargin, yPosA-ym)
        kptRecB = (xMargin+xSize, yPosB-20)
        if kpt == nextKpt:
            bdrRecA = (xMargin-8, yPosA-ym-4)
            kptRecA = (xMargin-8, yPosA-ym)
            bdrRecB = (xMargin+xSize+4, yPosB-ym+4)
            kptRecB = (xMargin+xSize, yPosB-ym)
            ym = 15
            cv.rectangle(winImg, bdrRecA, bdrRecB, KSHADOW_COLOR, -1)
            cv.rectangle(winImg, kptRecA, kptRecB, NEXTKPT_COLOR, -1)
        elif kpt in _pendingKpts[fid]:
            ym = 20
            cv.rectangle(winImg, kptRecA, kptRecB, PENDING_COLOR, -1)
        elif not FRAME_VALID_KPTS[kptIdx, fid]:
            ym = 20
            cv.rectangle(winImg, kptRecA, kptRecB, INVALID_COLOR, -1)
        else: # marked
            ym = 20
            if kptFrmError>MAX_KPT_ERROR or kptAggError>MAX_AGG_ERROR:
                anotKptColor = FAULTYK_COLOR
            else: anotKptColor = MARKEDK_COLOR
            cv.rectangle(winImg, kptRecA, kptRecB, anotKptColor, -1)

        # keypoint information text
        labPos = (xMargin+5, yPosA+5)
        if FRAME_VALID_KPTS[kptIdx, fid]:
            kptInfo = '{:>2}.  {:<3} {:>5.1f} {:>5.1f}'.\
                        format(kptAlpha, kpt, kptFrmError, kptAggError)
            cv.putText(winImg, kptInfo, labPos, FONT2, 0.9, KSHADOW_COLOR, 2, lineType=LINE1)
            cv.putText(winImg, kptInfo, labPos, FONT2, 0.9, KPT_COLORS[kpt], 1, lineType=LINE1)

    frmImage = highlight_next_keypoint_suggested_position(frmImage, fid, nextKpt)
    winImg[:FRM_HGT, :FRM_WDT] = frmImage
    return winImg

def previously_marked_keypoint(fid):
    frmPendingKpts = _pendingKpts[fid]
    if len(frmPendingKpts) > 0:
        nextKpt = frmPendingKpts[len(frmPendingKpts) - 1] # at end of list (stack)
        for i, kpt in enumerate(KPTS_STACK):
            if kpt == nextKpt:
                for j in range(i+1, N_KEYPOINTS):
                    indx = j % N_KEYPOINTS
                    prvKpt = KPTS_STACK[indx]
                    pKptid = KPTS_ID_INDX[prvKpt]
                    if FRAME_VALID_KPTS[pKptid, fid]:
                        return prvKpt # last valid kpt removed from stack
    else:
        for kpt in KPT_LIMB_ORDER:
            kptId = KPTS_ID_INDX[kpt]
            if FRAME_VALID_KPTS[kptId, fid]:
                return kpt
    return None

def remove_invalid_keypoints(fid):
    # remove invalid keypoints from pending list
    for kpt in _pendingKpts[fid].copy():
        kptIdx = KPTS_ID_INDX[kpt]
        if not FRAME_VALID_KPTS[kptIdx, fid]:
            _pendingKpts[fid].remove(kpt)

def annotate_frame(fid):
    global _iWinFrameID, _transMatrix, _scaleFactor, _contrastFtr, _brightenFtr, _brigContFtr
    _transMatrix = np.identity(3, dtype=np.float32) # initialize as identity matrix
    _scaleFactor = 1 # initialize with 1 which has no effect on scaling
    _contrastFtr = 1
    _brightenFtr = 0
    _brigContFtr = 0
    _iWinFrameID = fid
    altFrmImg = frame_visual_update(fid)
    display(WINDOW_NAME, altFrmImg, fid) # wait if last of triplet

def refresh_display(windowName, fid):
    altFrmImg = frame_visual_update(fid)
    display(windowName, altFrmImg, fid)

def display(windowName, displayImg, fid, wait=True):
    global _pendingKpts, _contrastFtr, _brightenFtr, _brigContFtr, _pauseForEvent, _mouseEventPack
    # show all flagged frames of a scan for marking
    _pauseForEvent = wait
    _mouseEventPack = [windowName, fid, 0]
    cv.imshow(windowName, displayImg)

    # continue to loop until a key-event turns off _pauseForEvent
    while _pauseForEvent:
        key = cv.waitKeyEx(1) # wait for a key press or mouse click
        # if mouse is clicked
        if _mouseEventPack[2] > 0:
            visualize_mouse_event(_mouseEventPack)

        # when to delete annotation for a particular keypoint
        if key in ALPHABET_ORD:
            idx = ALPHABET_ORD.index(key)
            delete_keypoint_marking(windowName, fid, ALPHABETS_ID[idx])
        # if the 'r' or 'delete' key is pressed, delete entire keypoint markings of frame
        if (key==ord("r") or key%255==46):
            reset_markings(windowName, fid)
        # if the 'backspace' key is pressed undo most recent (last) keypoint marking of frame
        elif (key==ord("u") or key%255==8):
            undo_previous_marking(windowName, fid)
        # if the 'b' or 'arrow up' key is pressed, increase contrast
        elif (key==ord("h") or key%255==38) and _contrastFtr<5:
            _contrastFtr = max(0, _contrastFtr + CONTR_INCR)
            refresh_display(windowName, fid)
        # if the 'd' or 'arrow down' key is pressed, decrease contrast
        elif (key==ord("l") or key%255==40) and _contrastFtr>0:
            _contrastFtr = max(0, _contrastFtr - CONTR_INCR)
            refresh_display(windowName, fid)
        # if the 'b' key is pressed, increase brightness
        elif key==ord("b") and -25<=_brightenFtr<=20:
            _brightenFtr += BRIGT_INCR
            _brigContFtr += BRCNT_INCR
            refresh_display(windowName, fid)
        # if the 'd' key is pressed, decrease brightness
        elif key==ord("d") and -20<=_brightenFtr<=25:
            _brightenFtr -= BRIGT_INCR
            _brigContFtr -= BRCNT_INCR
            refresh_display(windowName, fid)
        # if the 'n' or '-->' key is pressed, break from the loop and move to the next frame
        elif key==ord("n") or key%255==39:
            slide(fid, direction=1)
        # if the 'p' or '<--' key is pressed, break from loop and go back to previous frame
        elif key==ord("p") or key%255==37:
            slide(fid, direction=-1)
        # if the 'f' or 'enter' key is pressed, break from loop and go to the next scan
        elif key==ord("f") or key&255==13:
            next_scan(fid, forceSkip=True)
        # if the 'e' or 'esc' key is pressed, or window is closed
        # write dataframe to csv file before exiting the program
        elif key==ord("q") or key==27 or cv.getWindowProperty(windowName,0)<0:
            terminate_program(fid)
        # if the 'i' or 'space bar' key is pressed, display help message
        elif key==ord("i") or key%255==32:
            print (HELP_INSTRUCTION)

def initiate_3dpose_bundle_adjustment(ListOfKpts):
    global _projectedKptsPerFrm, _3dPose, _aggErrorPerKpt
    for kpt in ListOfKpts:
        kptIdx = KPTS_ID_INDX[kpt]
        areAnnotatedKpts = _annotatedKptsPerFrm[kptIdx][:,0]>=0
        if np.any(areAnnotatedKpts):
            # do bundle adjustment iff kpt has been truly marked in one or more valid frames
            projectedKpts, point3d, kptErrorPerFrm, frmsOfKpt = \
                project_keypoints_from_3d(kptIdx, _annotatedKptsPerFrm[kptIdx], areAnnotatedKpts)
            _projectedKptsPerFrm[kptIdx] = projectedKpts
            _3dPose[kptIdx] = point3d
            _aggErrorPerKpt[kptIdx] = np.sum(kptErrorPerFrm)
            kptErrorPerFrm = np.around(kptErrorPerFrm, 2)
            for idx, fid in enumerate(frmsOfKpt):
                if areAnnotatedKpts[idx]:
                    record_keypoint_meta(kpt, fid, None, None, kptErrorPerFrm[idx], onlyError=True)

def mouse_event(event, x, y, flags, param):
    # grab references to the global variables
    global _dictOfKpts, _pendingKpts, _transMatrix, _scaleFactor, _mouseEventPack, \
        _annotatedKptsPerFrm, _updatePoseImage
    fid = _iWinFrameID

    # if the left mouse button is clicked, record the starting (x, y) coordinates
    if event==cv.EVENT_LBUTTONDOWN and len(_pendingKpts[fid])>0:
        if (0<=x<=FRM_WDT and 0<=y<=FRM_HGT):
            kpt = _pendingKpts[fid].pop()
            kptIdx = KPTS_ID_INDX[kpt]
            xPos, yPos = transform_point(x, y, np.linalg.inv(_transMatrix))

            # make note of original marked point
            record_keypoint_meta(kpt, fid, xPos, yPos, None)

            # reconstruct keypoint's 3d position
            _updatePoseImage = True
            _annotatedKptsPerFrm[kptIdx][fid] = [xPos, yPos]
            initiate_3dpose_bundle_adjustment([kpt])
            if np.any(_3dPose[:,1:] >= 0): # Note: x may be <0
                plot_3d_pose(_3dPose, error=_aggErrorPerKpt)

            _mouseEventPack = [param, fid, 1] # 1: indicates keypoint marking

    # move to next frame
    elif event==cv.EVENT_RBUTTONDOWN:
        _mouseEventPack = [param, fid, 2] # 2: indicates move to next frame (or scan if complete)

    # zoom in or out on region if scroll down or up respectively
    elif event==cv.EVENT_MOUSEWHEEL:
        # EVENT_MOUSEWHEEL + and - values mean forward and backward scrolling, respectively
        zoom = False
        # zoom in
        if flags >= 0 and _scaleFactor < 10:
            zoom = True
            _scaleFactor = max(1, _scaleFactor + SCALE_INCR)
        # zoom out
        elif flags < 0 and _scaleFactor > 1:
            zoom = True
            _scaleFactor = max(1, _scaleFactor - SCALE_INCR)
        
        if zoom:
            # translation and scale transformation matrix.
            #  Notice: T*S*-T is equivalent to -T*S if scale factor(s) = 2
            s = _scaleFactor
            _transMatrix = np.float32([[s, 0, x-s*x], [0, s, y-s*y], [0, 0, 1]])
            _mouseEventPack = [param, fid, 3] # 2: indicates zooming

def iterate_over_scans(sampleMode, firstScan):
    global _dictOfKpts, _pendingKpts, _scanId, _curFrmImage, \
        _stayOnScan, _changeInScan, _moveDirection, _markerMetadata, \
        _annotatedKptsPerFrm, _projectedKptsPerFrm, _3dPose, _aggErrorPerKpt

    _moveDirection = 1
    _dictOfKpts = dict()
    _pendingKpts = dict()
    _3dPose = np.zeros((N_KEYPOINTS, 3), dtype=np.int32)
    _aggErrorPerKpt = np.zeros((N_KEYPOINTS), dtype=np.float32)
    _annotatedKptsPerFrm = np.zeros((N_KEYPOINTS, NUM_OF_FRAMES, 2), dtype=np.int32)
    _projectedKptsPerFrm = np.zeros((N_KEYPOINTS, NUM_OF_FRAMES, 2), dtype=np.int32)
    skipScan = False if firstScan=='' else True

    for index, row in _dfKpt.iterrows():
        if index<START_DF_IDX or index>STOP_DF_IDX:
            break  # skip scans not within start-stop-range
        subset = row['Subset']
        if pd.isna(subset) or subset not in SUBSETS:
            break  # skip over scans not in subsets
        recordStatus = row['Status']
        if pd.isna(recordStatus): recordStatus = EMPTY_CELL
        if recordStatus not in sampleMode:
            break  # skip over scans not in record status type

        _scanId = row['scanID']
        scanNpIdx = row['npIndex']
        if skipScan and firstScan!=_scanId: break
        else: skipScan = False

        # reset variables to default
        _dictOfKpts.clear()
        _pendingKpts.clear()
        _3dPose[:,:] = -1
        _aggErrorPerKpt[:] = 0.
        _annotatedKptsPerFrm[:,:,:] = -1
        _projectedKptsPerFrm[:,:,:] = -1

        # parse 3d-pose
        cellEntry = row['3dPose']
        if not pd.isna(cellEntry):
            # read in 3dPose
            pose3dDict = ast.literal_eval(cellEntry)
            for kpt in KEYPOINTS_ID:
                kptIdx = KPTS_ID_INDX[kpt]
                _3dPose[kptIdx] = pose3dDict[kpt][:3]
                _aggErrorPerKpt[kptIdx] = pose3dDict[kpt][3] # last for 3d-pose

            if np.any(_3dPose[:,1:] >= 0): # Note: -256<x<256, x may be <0
                plot_3d_pose(_3dPose, error=_aggErrorPerKpt)

        # iterate over frames
        for fid in range(NUM_OF_FRAMES):
            # initialize pending stack
            _pendingKpts[fid] = KPTS_STACK.copy()
            remove_invalid_keypoints(fid)

            # remove already marked keypoints from pending list
            cellEntry = row['Frame{}'.format(fid)] # dictionary or EMPTY_CELL
            if pd.isna(cellEntry):
                # initialize default values for annotations
                frmKptsDict = dict()
                for kpt in KEYPOINTS_ID:
                    kptIdx = KPTS_ID_INDX[kpt]
                    if FRAME_VALID_KPTS[kptIdx, fid]:
                        frmKptsDict[kpt] = list(K_DEFAULT)  # Default values for x,y,e=[-1,-1,0]
                _dictOfKpts[fid] = frmKptsDict
            else:
                # load annotations from record
                frmKptsDict = ast.literal_eval(cellEntry)
                # remove keypoints with annotations from pending list
                for kpt in frmKptsDict.keys():
                    if kpt in _pendingKpts[fid]:
                        _pendingKpts[fid].remove(kpt)
                # fill in dummy/placeholder info for keypoints yet to be annotated
                for kpt in _pendingKpts[fid]:
                    frmKptsDict[kpt] = list(K_DEFAULT)  # Default values for x,y,e=[-1,-1,0]
                _dictOfKpts[fid] = frmKptsDict

        print('{:>4}. scanID: {}'.format(index, _scanId))
        cnt = 0
        t0 = time.time()
        _stayOnScan = True
        _changeInScan = False
        while _stayOnScan:
            fid = cnt % NUM_OF_FRAMES
            _curFrmImage = get_npy_frame_image(_npyFile, scanNpIdx, fid)
            if _curFrmImage is not None:
                annotate_frame(fid)
            elif cnt==(_moveDirection*NUM_OF_FRAMES):
                _stayOnScan = False  # break out of infinte-loop
            cnt += _moveDirection # 1 or -1
        t1 = time.time()
        os.remove(POSE3D_FIG_PATH)
        if _changeInScan:
            _markerMetadata['totalTime'] += t1-t0


if __name__ == "__main__":
    global _dfKpt, _npyFile, _updateSinceLastAutoSave, _updatePoseImage, _markerMetadata, \
            HELP_INSTRUCTION, WINDOW_NAME, FONT1, FONT2, LINE1, LINE2
    args = runtime_args()
    mode = args.sampleMode
    startScan = args.scanSample
    HELP_INSTRUCTION = get_program_instructions()

    # load or set user/marker metadata information tracker
    random.seed(MARKER_NAME)
    markerID = '{}{}'.format(MARKER_NAME, random.randint(10,99))
    KPTS_CSV_FILE = KPTS_CSV_FILE.format(markerID)
    METADATA_PATH = METADATA_PATH.format(markerID)
    if os.path.exists(METADATA_PATH):
        print("\n Welcome back {}!\n\n Here's a reminder of the program instructions: {}".
              format(MARKER_NAME, HELP_INSTRUCTION))
        with open(METADATA_PATH, 'rb') as file_handle:
            _markerMetadata = pickle.load(file_handle)
    else:
        print("\n Welcome {}!\n\n Here are the program instructions: {}".
              format(MARKER_NAME, HELP_INSTRUCTION))
        _markerMetadata = {'markerID':markerID, 'completedScans':0,
                           'totalTime':0, 'totalError':np.zeros(N_KEYPOINTS)}

    # load numpy data and keypoint annotation record and initialize default values
    _npyFile = np.load(IMAGES_NPFILE, mmap_mode='r')
    if os.path.exists(KPTS_CSV_FILE):
        _dfKpt = pd.read_csv(KPTS_CSV_FILE)
    else: _dfKpt = pd.read_csv(TEMP_CSV_FILE)
    _updateSinceLastAutoSave = False
    _updatePoseImage = True
    
    # create windows and set mouse event listeners
    WINDOW_NAME = 'TSA Scan Keypoints/Joint Annotations'
    FONT1 = cv.FONT_HERSHEY_COMPLEX_SMALL
    FONT2 = cv.FONT_HERSHEY_PLAIN
    LINE1 = cv.LINE_AA
    LINE2 = cv.FILLED
    define_windows([WINDOW_NAME], 1, WINDOW_SHAPE[1], WINDOW_SHAPE[0],
                   yStart=WINDOW_TOP_LEFT[1], xStart=WINDOW_TOP_LEFT[0])
    # set mouse callback function for window
    cv.setMouseCallback(WINDOW_NAME, mouse_event, param=WINDOW_NAME)

    # thread to recursively call save function after every 10 minutes
    periodic_save_thread = threading.Thread(target=periodic_save)
    periodic_save_thread.daemon = True
    periodic_save_thread.start()

    iterate_over_scans(RUN_MODES[mode], startScan)
    if _updateSinceLastAutoSave: save_progress()
    cv.destroyAllWindows() # close all open windows
    os.remove(POSE3D_FIG_PATH)
    plt.close()
