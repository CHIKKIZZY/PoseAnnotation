# -*- coding: utf-8 -*-
# @Time    : 1/23/2021 2:11 PM
# @Author  : Lawrence A.
# @Email   : lamadi@hawk.iit.edu
# @File    : reconstruct.py
# @Software: reconstruct 3d pose of keypoints using bundle adjustment

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

sys.path.append('../')
from default import DEFAULT_3DPOSE, AX_SP, POSE3D_FIG_PATH
from default import N_KEYPOINTS, NUM_OF_FRAMES, LS_ALGO_STATUS, FRM_WDT, FRM_HGT
from default import KPTS_ID_INDX, KPTS_INDX_ID, FRAME_VALID_KPTS, LIMB_KPTS_PAIRS


def plot_3d_pose(points3d, error=None, display=False):
    points3d = np.int32(points3d)
    xKpts = points3d[:, 0]
    yKpts = points3d[:, 1]
    zKpts = points3d[:, 2]

    AX_SP.clear()
    AX_SP.set(xlim=(-FRM_WDT//2, FRM_WDT//2), zlim=(FRM_HGT, 0), ylim=(0, FRM_WDT))
              #xlabel='x', ylabel='z', zlabel='y', title='3d Pose')
    # Turn off tick labels
    AX_SP.set_yticklabels([])
    AX_SP.set_xticklabels([])
    AX_SP.set_zticklabels([])

    # Draw skeletal limbs (lines between keypoints)
    for limb, kptPair in LIMB_KPTS_PAIRS.items():
        kptA, kptB = kptPair
        if kptA[0]=='R' or kptB[0]=='R': color = 'olive'
        elif kptA[0]=='L' or kptB[0]=='L': color = 'purple'
        else: color = 'black'
        kptAIdx, kptBIdx = KPTS_ID_INDX[kptA], KPTS_ID_INDX[kptB]
        if yKpts[kptAIdx]>=0 and yKpts[kptBIdx]>=0:
            # assert (isinstance(kptA, int) and isinstance(kptB, int))
            zline = [yKpts[kptAIdx], yKpts[kptBIdx]]
            xline = [xKpts[kptAIdx], xKpts[kptBIdx]]
            yline = [zKpts[kptAIdx], zKpts[kptBIdx]]
            AX_SP.plot(xline, yline, zline, color, zdir='z')

    # Data for three-dimensional keypoints
    validKptIndexes = np.argwhere(yKpts>=0).flatten()
    zData = yKpts[validKptIndexes]
    xData = xKpts[validKptIndexes]
    yData = zKpts[validKptIndexes]
    error = error[validKptIndexes]
    AX_SP.scatter3D(xData, yData, zData, c=error, cmap='summer') #RdYlGn

    if display: plt.pause(0.001)
    else: plt.savefig(POSE3D_FIG_PATH)

def keypoint_error_magnitude(error):
    # Note: error is the x-y deviation between the adjusted/optimized 3d-kpt
    # projected to each 2d frame and the actual 2d-kpt in each frame
    # compute euclidean distance of kpt deviation in each frame
    kptErrorPerFrm = np.square(error).reshape((-1, 2))  # (nFrms, 2)
    kptErrorPerFrm = np.sum(kptErrorPerFrm, axis=-1)
    kptErrorPerFrm = np.sqrt(kptErrorPerFrm)
    return kptErrorPerFrm

def error_per_keypoint(error, kptIndiciesList):
    error = np.square(error).reshape((-1, 2))
    print(error)
    pntWgt = np.zeros(len(kptIndiciesList))
    for i, kptIndicies in enumerate(kptIndiciesList):
        kptError = np.sum(error[kptIndicies, :]) / 2
        print(kptError)
        pntWgt[i] = kptError
    return pntWgt / (np.max(pntWgt) + 1e-7)

def get_keypoint_indicies_list(pointIndices, nPoints):
    '''Scale confidence scores to (0,1] per keypoint then return square root'''
    kptIndiciesList = list()
    for kptIdx in range(nPoints):
        kptIndices = np.argwhere(pointIndices == kptIdx)
        kptIndiciesList.append(kptIndices)
    return kptIndiciesList

def circular_translate_vec(thetaDeg, radius):
    diameter = 2 * radius
    assert (0 <= thetaDeg < 360) # theta must be in degrees
    alphaDeg = thetaDeg if thetaDeg <= 180 else 360 - thetaDeg
    assert (0 <= alphaDeg <= 180)
    alphaRad = np.deg2rad(alphaDeg)
    trsMagn = np.sqrt((2 * (radius**2)) - (2 * (radius**2)) * np.cos(alphaRad))
    assert (0 <= trsMagn <= diameter)
    phiMagn = (180 - alphaDeg) / 2
    assert (0 <= phiMagn <= 90)
    phiSign = -1 if (180 - thetaDeg) < 0 else 1
    assert (phiSign == 1 or phiSign == -1)
    phiDeg = phiSign * phiMagn
    assert (-90 <= phiDeg <= 90)
    phiRad = np.deg2rad(phiDeg)
    cosPhi = np.cos(phiRad)
    assert (0 <= cosPhi <= 1)
    zTrans = trsMagn * cosPhi
    assert (0 <= zTrans <= diameter)
    sinPhi = np.sin(phiRad)
    assert (-1 <= sinPhi <= 1)
    xTrans = trsMagn * sinPhi
    assert (-diameter <= xTrans <= diameter)
    assert (-radius <= xTrans <= radius)
    transVec = np.array([xTrans, 0, zTrans]) #np.transpose(np.array([[xTrans, 0, zTrans]]))
    return transVec

def rotate_about_yaxis_mtx(thetaDeg):
    assert (0 <= thetaDeg < 360)
    thetaRad = np.deg2rad(thetaDeg)
    cosTheta = np.cos(thetaRad)
    sinTheta = np.sin(thetaRad)
    rotMtx = np.zeros((3, 3), dtype=np.float32)
    rotMtx[0, :] = [cosTheta, 0, sinTheta]
    rotMtx[1, 1] = 1
    rotMtx[2, :] = [-sinTheta, 0, cosTheta]
    return rotMtx

def wld2cam_transformation_matrix(angleDeg, radius):
    translateVec = circular_translate_vec(angleDeg, radius)
    rotateMtx = rotate_about_yaxis_mtx(angleDeg)
    transformMtx = np.zeros((3, 4), dtype=np.float32)
    transformMtx[:3, :3] = rotateMtx
    transformMtx[:3, 3] = rotateMtx.dot(-translateVec)
    return transformMtx

def get_annotated_keypoints(scanid, df):
    kptsMeta = np.zeros((N_KEYPOINTS, NUM_OF_FRAMES, 2), dtype=np.float32)
    isMarkedKpts = np.zeros((N_KEYPOINTS, NUM_OF_FRAMES), dtype=np.bool)
    for fid in range(NUM_OF_FRAMES):
        columnName = 'Frame{}'.format(fid)
        cellEntry = df.loc[df['scanID']==scanid, columnName].values[0]
        if not pd.isna(cellEntry):
            frmKptsMeta = eval(cellEntry)  # eval() or ast.literal_eval()
            for kptIdx in range(N_KEYPOINTS):
                kpt = KPTS_INDX_ID[kptIdx]
                kptsMeta[kptIdx, fid, :] = frmKptsMeta[kpt]
                isMarkedKpts[kptIdx, fid] = True
    return kptsMeta, isMarkedKpts

def valid_observations(isMarkedKptsSet, nKpts, kptIdx=None):
    assert (nKpts!=1 or kptIdx is not None)  # p-->r
    maxObservations = nKpts * NUM_OF_FRAMES

    # 3D points initial estimates should come from best aps keypoints
    kptsValidFrms = FRAME_VALID_KPTS[kptIdx] if nKpts==1 else FRAME_VALID_KPTS
    useKptsFromFrms = np.logical_and(isMarkedKptsSet, kptsValidFrms)
    nObservations = np.sum(np.int32(useKptsFromFrms))
    assert (0<=nObservations<=maxObservations)

    # note frames from which keypoints are taken
    frmsOfEachKpt = list()
    for kptIdx in range(nKpts):
        frmsOfKpt = list()
        for fid in range(NUM_OF_FRAMES):
            if useKptsFromFrms[fid]:
                frmsOfKpt.append(fid)
        frmsOfEachKpt.append(frmsOfKpt)

    return nObservations, frmsOfEachKpt

def define_camera_extrinsic_params(nCameras, radius):
    cameraParams = np.empty((nCameras, 3, 4), dtype=np.float32)
    angleStep = 360 / nCameras
    for camIdx in range(nCameras):
        angleDeg = camIdx * angleStep
        assert (0 <= angleDeg <= 360)
        extTfmMtx = wld2cam_transformation_matrix(angleDeg, radius)
        cameraParams[camIdx] = extTfmMtx
    return cameraParams

def organize_keypoints(frmsAnnotatedKpts, isMarkedKptsSet, kptIdx, zDepth):
    # from a single annotated keypoint set (multiple frames)
    # Note. For orthographic projection, lens focal length is infinity
    nObservations, frmsOfEachKpt = valid_observations(isMarkedKptsSet, 1, kptIdx)
    nCameras = max(16, NUM_OF_FRAMES)

    cameraIndices = np.empty(nObservations, dtype=np.int32)
    pointIndices = np.empty(nObservations, dtype=np.int32)
    points2d = np.empty((nObservations, 2), dtype=np.float32)
    idx = 0

    for fid in range(NUM_OF_FRAMES):
        x, y = frmsAnnotatedKpts[fid]
        if FRAME_VALID_KPTS[kptIdx][fid] and isMarkedKptsSet[fid]:
            if nCameras==16 or NUM_OF_FRAMES==64: cameraIndex = fid
            else: cameraIndex = (fid * 4) % 64 # nCameras==64 and NUM_OF_FRAMES==16
            cameraIndices[idx] = cameraIndex
            pointIndices[idx] = 0
            points2d[idx] = [x, y]
            idx += 1
    assert (idx == nObservations)

    radius = zDepth / 2
    cameraParams = define_camera_extrinsic_params(nCameras, radius)

    points3d = np.empty((1, 3), dtype=np.float32)
    # Assumes perfect alignment, relative to ref. camera frame coordinate.
    kid = KPTS_INDX_ID[kptIdx]
    x, y, z = DEFAULT_3DPOSE[kid]
    x = x - 256
    points3d[0] = [x, y, z]

    assert (np.all(0 <= points2d[:, 0]) and np.all(points2d[:, 0] < 512))
    assert (np.all(0 <= points2d[:, 1]) and np.all(points2d[:, 1] < 660))
    assert (np.all(-256 <= points3d[:, 0]) and np.all(points3d[:, 0] < 256))
    assert (np.all(0 < points3d[:, 1]) and np.all(points3d[:, 1] < 660))
    assert (np.all(points3d[:, 2] == 256))
    return cameraParams, points3d, cameraIndices, pointIndices, points2d, frmsOfEachKpt[0]

def organize_all_scan_keypoints(scanAnnotatedKpts, isMarkedKptsSet, zDepth):
    # from a multiple set of annotated keypoints of a scan
    # Note. For orthographic projection, lens focal length is infinity
    # scanAnnotatedKpts: shape: (14:kpts, 16/64:frames, 2:(x,y))
    assert (scanAnnotatedKpts.dtype == np.float32)
    assert (scanAnnotatedKpts.shape == (N_KEYPOINTS,16,2) or (N_KEYPOINTS,64,2))
    nObservations, frmsOfEachKpt = valid_observations(isMarkedKptsSet, nKpts=N_KEYPOINTS)
    nCameras = max(16, NUM_OF_FRAMES)

    cameraIndices = np.empty(nObservations, dtype=np.int32)
    pointIndices = np.empty(nObservations, dtype=np.int32)
    points2d = np.empty((nObservations, 2), dtype=np.float32)
    idx = 0

    for kptIdx in range(N_KEYPOINTS):
        pointIndex = kptIdx
        for fid in range(NUM_OF_FRAMES):
            x, y = scanAnnotatedKpts[kptIdx][fid]
            if FRAME_VALID_KPTS[kptIdx][fid] and isMarkedKptsSet[kptIdx][fid]:
                if nCameras==16 or NUM_OF_FRAMES==64: cameraIndex = fid
                else: cameraIndex = (fid * 4) % 64 # nCameras==64 and NUM_OF_FRAMES==16
                cameraIndices[idx] = cameraIndex
                pointIndices[idx] = pointIndex
                points2d[idx] = [x, y]
                idx += 1
    assert (idx == nObservations)

    radius = zDepth / 2
    cameraParams = define_camera_extrinsic_params(nCameras, radius)

    points3d = np.empty((N_KEYPOINTS, 3), dtype=np.float32)
    points3d[:, 2] = radius # Assumes perfect alignment, relative to ref. camera frame coordinate.
    for kptIdx in range(N_KEYPOINTS):
        x, y = scanAnnotatedKpts[kptIdx][0]
        x = x - 256
        points3d[kptIdx, :2] = [x, y]

    assert (np.all(0 <= points2d[:, 0]) and np.all(points2d[:, 0] < 512))
    assert (np.all(0 <= points2d[:, 1]) and np.all(points2d[:, 1] < 660))
    assert (np.all(-256 <= points3d[:, 0]) and np.all(points3d[:, 0] < 256))
    assert (np.all(0 < points3d[:, 1]) and np.all(points3d[:, 1] < 660))
    assert (np.all(points3d[:, 2] == 256))
    return cameraParams, points3d, cameraIndices, pointIndices, points2d

def orthographic_project(points3dh, cameraParams, nObservations):
    """Convert 3-D points to 2-D by projecting onto images."""
    # points3dh.shape = (nObservations, 4)
    # cameraParams.shape = (nObservations, 3, 4)
    extTfmMatrices = cameraParams #.reshape(-1, 3, 4)
    indexes = np.arange(nObservations)
    assert (nObservations == extTfmMatrices.shape[0])
    camKpts3d = np.dot(extTfmMatrices, points3dh)
    camKpts3d = camKpts3d[indexes, :, indexes]
    pointsProj = camKpts3d[:, :2] + [256, 0]
    return pointsProj

def res_func(params, cameraParams, nPoints, nObservations,
             cameraIndices, pointIndices, points2d):
    """
    Compute residuals.
    `params` contains 3-D coordinates only.
    """
    points3d = params.reshape((nPoints, 3))
    points3dh = np.append(points3d, np.ones((nPoints, 1)), axis=1)
    # duplicates 3d points and camera parameters to match number of 2d point observations
    dupPoints3d = points3dh[pointIndices]
    dupPoints3d = np.swapaxes(dupPoints3d, 0, 1)
    dupCameraParams = cameraParams[cameraIndices]
    pointsProj = orthographic_project(dupPoints3d, dupCameraParams, nObservations)
    return (pointsProj - points2d).ravel()

def points_only_bundle_adjustment_sparsity(nPoints, pointIndices):
    size = pointIndices.size
    m = size * 2
    n = nPoints * 3
    A = lil_matrix((m, n), dtype=np.int32)

    i = np.arange(size)
    for s in range(3):
        A[2 * i, pointIndices * 3 + s] = 1
        A[2 * i + 1, pointIndices * 3 + s] = 1
    return A

def project_3dpoint_to_2dkeypoints(point3d, radius=256):
    wldKpt3dh = np.ones((4, 1), dtype=np.float32)
    angleStep = 360 / NUM_OF_FRAMES
    projectedKpts = np.zeros((NUM_OF_FRAMES, 2), np.float32)

    for fid in range(NUM_OF_FRAMES):
        angleDeg = fid * angleStep
        extTfmMtx = wld2cam_transformation_matrix(angleDeg, radius)

        wldKpt3d = point3d[:, np.newaxis]
        wldKpt3dh[:3, :] = wldKpt3d
        camKpt3d = extTfmMtx.dot(wldKpt3dh)
        xCam, yCam, zCam = np.around(camKpt3d, 0)
        xImg = int(xCam + 256)
        yImg = int(yCam)
        #***assert (0 <= xImg < 512)
        #***assert (0 <= yImg < 660)
        projectedKpts[fid] = [xImg, yImg]

    return projectedKpts

def project_all_3dpoints_to_2dkeypoints(points3d, error, radius=256):
    wldKpt3dh = np.ones((4, 1), dtype=np.float32)
    angleStep = 360 / NUM_OF_FRAMES
    frmProjectedKpts = list()

    for fid in range(NUM_OF_FRAMES):
        projectedKpts = dict()
        angleDeg = fid * angleStep
        extTfmMtx = wld2cam_transformation_matrix(angleDeg, radius)
        for kptIdx in range(N_KEYPOINTS):
            kpt = KPTS_INDX_ID[kptIdx]
            wldKpt3d = points3d[kptIdx][:, np.newaxis]
            wldKpt3dh[:3, :] = wldKpt3d
            camKpt3d = extTfmMtx.dot(wldKpt3dh)
            xCam, yCam, zCam = np.around(camKpt3d, 0)
            xImg = int(xCam + 256)
            yImg = int(yCam)
            conf = error[kptIdx]
            assert (0 <= xImg < 512)
            assert (0 <= yImg < 660)
            assert (0 <= conf <= 1)
            projectedKpts[kpt] = (xImg, yImg, conf)
        frmProjectedKpts.append(projectedKpts)

    return frmProjectedKpts


def project_keypoints_from_3d(kptIdx, kpt2dPoints, isMarkedKpts, verbose=0, showInfo=False):
    frmValidObservations = np.sum(np.int32(FRAME_VALID_KPTS[kptIdx]))

    # initial estimated keypoints
    cameraParams, points3d, cameraIndices, pointIndices, points2d, frmsOfKpt = \
        organize_keypoints(kpt2dPoints, isMarkedKpts, kptIdx, zDepth=FRM_WDT)

    nObservations = points2d.shape[0]
    assert (frmValidObservations >= nObservations)
    nCameras = cameraParams.shape[0]
    nPoints = points3d.shape[0]
    x0 = points3d.ravel()

    # Bundle adjustment optimization
    A = points_only_bundle_adjustment_sparsity(nPoints, pointIndices)
    #kptIndicesList = [pointIndices]
    t0 = time.time()
    res = least_squares(res_func, x0,
                jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf',
                args=(cameraParams, nPoints, nObservations, cameraIndices, pointIndices, points2d))
    t1 = time.time()

    adjPoint3d = res.x.reshape((3,))
    kptErrorPerFrm = keypoint_error_magnitude(res.fun)
    projectedKpts = project_3dpoint_to_2dkeypoints(adjPoint3d)

    if showInfo:
        error = np.sum(np.square(res.fun)) / 2
        residual_vec = np.int32(np.around(res.fun.reshape((nObservations, 2)), 0))
        print('\nresiduals at solution:\nx\n{}\ny\n{}\nshape: {}, square sum: {}'
              .format(residual_vec[:, 0], residual_vec[:, 1], residual_vec.shape, error))
        print('\nerror at solution: {}'.format(res.cost))
        print('\nalgorithm terminated because, {}'.format(LS_ALGO_STATUS[res.status]))
        print("nCameras: {}".format(nCameras))
        print("nPoints: {}".format(nPoints))
        n = np.size(cameraParams) + np.size(points3d)
        m = np.size(points2d)
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        print('3d points\n', np.int32(np.around(adjPoint3d, 0)))
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

    return projectedKpts, adjPoint3d, kptErrorPerFrm, frmsOfKpt


def project_all_keypoints_from_3d(scanid, df, verbose=0, showInfo=False, plot3dpose=True):
    frmValidObservations = np.sum(np.int32(FRAME_VALID_KPTS))

    # initial estimated keypoints
    annotatedKpts, isMarkedKpts = get_annotated_keypoints(scanid, df)
    cameraParams, points3d, cameraIndices, pointIndices, points2d = \
        organize_all_scan_keypoints(annotatedKpts, isMarkedKpts, zDepth=FRM_WDT)

    nObservations = points2d.shape[0]
    assert (frmValidObservations >= nObservations)
    nCameras = cameraParams.shape[0]
    nPoints = points3d.shape[0]
    x0 = points3d.ravel()

    # Bundle adjustment optimization
    A = points_only_bundle_adjustment_sparsity(nPoints, pointIndices)
    kptIndicesList = get_keypoint_indicies_list(pointIndices, nPoints)
    t0 = time.time()
    res = least_squares(res_func, x0,
                jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf',
                args=(cameraParams, nPoints, nObservations, cameraIndices, pointIndices, points2d))
    t1 = time.time()

    adjPoints3d = res.x.reshape((nPoints, 3))
    kptsError = error_per_keypoint(res.fun, kptIndicesList)
    frmProjectedKpts = project_all_3dpoints_to_2dkeypoints(adjPoints3d, kptsError)

    if showInfo:
        error = np.sum(np.square(res.fun)) / 2
        residual_vec = np.int32(np.around(res.fun.reshape((nObservations, 2)), 0))
        print('\nresiduals at solution:\nx\n{}\ny\n{}\nshape: {}, square sum: {}'
              .format(residual_vec[:, 0], residual_vec[:, 1], residual_vec.shape, error))
        print('\nerror at solution: {}'.format(res.cost))
        print('\nalgorithm terminated because, {}'.format(LS_ALGO_STATUS[res.status]))
        print("nCameras: {}".format(nCameras))
        print("nPoints: {}".format(nPoints))
        n = np.size(cameraParams) + np.size(points3d)
        m = np.size(points2d)
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))
        print('3d points\n', np.int32(np.around(adjPoints3d, 0)))
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

    if plot3dpose:
        plot_3d_pose(adjPoints3d, error=kptsError)

    return frmProjectedKpts, adjPoints3d