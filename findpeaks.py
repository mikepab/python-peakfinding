# Mike Pablo 2017 =================================================
# findpeaks(y) returns peak height, indices, prominences, and FWHM.
# Note that this function, unlike MATLAB's, is not written to allow for
# irregular sampling in x. If your sampling is irregular, you can interpolate
# before applying findpeaks().  
#
# From MATLAB:
# "The prominence of a peak measures how much the peak stands out due to
# its intrinsic height and its location relative to other peaks. An isolated
# peak can be more prominent than one that is higher but is an otherwise un-
# remarkable member of a tall range."
# 
# See https://www.mathworks.com/help/signal/ref/findpeaks.html
# for more detail.


import numpy as np
# import warnings
# warnings.filterwarnings("ignore",category =RuntimeWarning)

# peaks = findpeaks(y) returns peak height, indices, prominences, and FWHMs.
# Access with peaks['height'], peaks['idx'], peaks['prom'], and peaks['FWHM']
def findpeaks(y):
    # get all peaks
    allpeaks = getAllPeaks(y)
    
    # find extents of peaks
    peaks = findExtents(y,allpeaks['iPk'],allpeaks['iInf'],allpeaks['iInflect'])
    
    return {'height':y[peaks['iPk']],'idx':peaks['iPk'],\
            'prom':(y[peaks['iPk']]-peaks['bPk']),\
            'FWHM':peaks['wxPk']}
    
# ret = getAllPeaks(y) returns finite/infinite peaks 
# and inflection point indices.
# Access with ret['iPk'], ret['iInf'], and ret['iInflect'].
def getAllPeaks(y):
    # check for infinite peaks. throw a warning or exception if found?
    iInf = np.isinf(y)
    
    if np.any(iInf):
        warnings.warn("Infinite value encountered during findpeaks")
    
    # remove +inf values
    y[iInf] = np.nan
    
    # determine local maxima and inflection points w/o the +infs
    maxData = findLocalMaxima(y)
    return {'iPk':maxData['iPk'],'iInf':iInf,'iInflect':maxData['iInflect']}


# ret = findLocalMaxima(yTemp) returns peak and inflection point indices.
# Access with ret['iPk'] and ret['iInflect'].
def findLocalMaxima(yTemp):
    # pad ends by NaN, construct indexing vector
    yTemp = np.insert(yTemp,[0,len(yTemp)],np.nan)
    iTemp = np.array(range(0,len(yTemp)))
    
    #Keep only first of any adjacent pairs of equal values (incls. nans)
    iNeq = np.flatnonzero(yTemp[0:-1] != yTemp[1:]) + 1
    iNeq = np.insert(iNeq,0,0)

    
    iTemp = iTemp[iNeq]
    
    yTemp_sub = yTemp[iTemp]
    
    # take sign of first sample derivative
    s = np.sign(yTemp_sub[1:] - yTemp_sub[0:-1])

    # Find the local maxima
    s_diff = s[1:] - s[0:-1]
    
    iMax = 1 + np.flatnonzero(s_diff<0)

    # Find all transitions from rising to falling or to NaN
    iAny = 1 + np.flatnonzero(s[0:-1] != s[1:])

    # Index into original index vector without NaN bookend
    iInflect = iTemp[iAny]-1
    iPk = iTemp[iMax]-1
    
    return {'iPk':iPk,'iInflect':iInflect}

# ret = findExtents(y,iPk,iInf,iInfl) returns peak indices,
# base peak heights, the left and right indices of the bases, and
# the FWHM of the peaks.
# Access with ret['iPk'], ret['bPk'], ret['bxPk'], ret['byPk'], and
# ret['wxPk'].
def findExtents(y,iPk,iInf,iInfl):
    # remove +inf values
    yFinite = y
    yFinite[iInf] = np.nan
    
    # Get base left and right indices from each prominence's base
    peakbase = getPeakBase(yFinite,iPk,iInfl)    
    
    # getPeakWidth. Assumes you want half-prominence.
    peakwidth = getPeakWidth(yFinite,iPk,peakbase['bPk'],\
                             peakbase['iLB'],peakbase['iRB'])
    
    # We ignore the infinite peak data. Return from finite data.
    return {'iPk':iPk,'bPk':peakbase['bPk'],\
            'bxPk':peakbase['iLB'],'byPk':peakbase['iRB'],\
            'wxPk':peakwidth}
    
    

# ret = getPeakBase(yTemp,iPk,iInflect) returns peak base heights
# with corresponding left and right base indices.
# Access with ret['bPk'], ret['iLB'] and ret['iRB'].
def getPeakBase(yFinite,iPk,iInflect):
    # Get left base and saddle point
    LHSdata = getLeftBase(yFinite,iPk,iInflect)
    
    # Get right base and saddle point by inverting the search direction
    RHSdata = getLeftBase(yFinite,np.flip(iPk,0),\
                         np.flip(iInflect,0))
    
    # Peaks were analyzed in opposite order, so need to re-flip
    iRightBase = np.flip(RHSdata['iBase'],0)
    iRightSaddle = np.flip(RHSdata['iSaddle'],0)
    
    # Get higher of left/right bases for each peak.
    peakBase = np.maximum(yFinite[LHSdata['iBase']],yFinite[iRightBase])
    
    return {'bPk':peakBase,'iLB':LHSdata['iBase'],'iRB':iRightBase}

# ret = getLeftBase(yTemp,iPk,iInflect) returns peak base indices
# and saddle indices to the LEFT of the indices iPk,iInflect in yTemp.
# Access with ret['iBase'] and ret['iSaddle'].
def getLeftBase(yTemp,iPk,iInflect):
    # Pre-initialize output base and saddle indices
    iBase = np.zeros(np.shape(iPk))
    iSaddle = np.zeros(np.shape(iPk))
    
    # Generate vectors, which will store most recently encountered peaks
    # in order of height.
    peak = np.zeros(np.shape(iPk))
    valley = np.zeros(np.shape(iPk))
    iValley = np.zeros(np.shape(iPk))
    
    # Python indexing versus MATLAB indexing
    n = -1
    i = 0
    j = 0
    k = 0
    
    # iterate through all peaks
    while k < len(iPk):
        # walk through inflections until you find a peak
        while iInflect[i] != iPk[j]:
            v = yTemp[iInflect[i]]
            iv = iInflect[i]
            # Encountered a border, start over
            if np.isnan(v):
                n = -1
            else:
                # Ignore previously-stored peaks with valley larger than this
                while n > 0 and valley[n] > v:
                    n = n - 1
            i = i + 1
        # take the peak
        p = yTemp[iInflect[i]]
              
        # keep smallest valley of all smaller peaks
        while n >= 0 and peak[n] < p:
            if valley[n] < v:
                v = valley[n]
                iv = iValley[n]
            n = n - 1
                
        # record "saddle" valleys in between equal-height peaks
        isv =iv
        
        # keep seeking smaller valleys until you find a LARGER peak
        while n >= 0 and peak[n] <= p:
            if valley[n] < v:
                v = valley[n]
                iv = iValley[n]
            n = n - 1
        
        # record new peak, save index of valley into base and saddle vectors
        n = n + 1
        peak[n] = p
        valley[n] = v
        iValley[n] = iv
        
        if iInflect[i] == iPk[k]:
            iBase[k] = iv
            iSaddle[k] = isv
            k = k + 1
            
        i = i + 1
        j = j + 1
        
    iBase = (np.round(iBase)).astype(int)
    iSaddle = (np.round(iSaddle)).astype(int)
    return {'iBase':iBase,'iSaddle':iSaddle}

# FWHM = getLeftBase(yTemp,iPk,bPk,iLB,iRB) returns peak FWHMs.
def getPeakWidth(yTemp,iPk,bPk,iLB,iRB):
    base = bPk
    FWHM = getHalfMaxBounds(yTemp,iPk,base,iLB,iRB)
    
    return abs(FWHM[:,1]-FWHM[:,0])

# bounds = getHalfMaxBounds. bounds is a n x 2 NumPy array, where n is
# the number of peaks.
def getHalfMaxBounds(y,iPk,base,iLB,iRB):
    bounds = np.zeros( (len(iPk),2) )
    x = range(0,len(y))
    
    # Interpolate both left and right bounds, clamping at borders.
    for i in range(0,len(iPk)):

        # Compute desired reference level @ half-prominence
        refHt = (y[iPk[i]] + base[i])/2
        
        # Compute index of left intercept at half-max
        iLeft = findLeftIntercept(y,iPk[i],iLB[i],refHt)
        
        if iLeft < iLB[i]:
            xLeft = x[iLB[i]]
        else:
            xLeft = linterp(x[iLeft],x[iLeft+1],y[iLeft],y[iLeft+1],\
                            y[iPk[i]],base[i])
                
        # Compute index of right intercept at half-max
        iRight = findRightIntercept(y,iPk[i],iRB[i],refHt)
        
        if iRight > iRB[i]:
            xRight = x[iRB[i]]
        else:
            xRight = linterp(x[iRight],x[iRight-1],y[iRight],y[iRight-1],\
                            y[iPk[i]],base[i])
        
        bounds[i,:] = np.array( (xLeft,xRight) )
    
    return bounds
        
def findLeftIntercept(y,idx,borderIdx,refHeight):
    # decrement index until you pass under ref ht, or pass border
    while idx >= borderIdx and y[idx] > refHeight:
        idx = idx - 1
    
    return idx

def findRightIntercept(y,idx,borderIdx,refHeight):
    # increment index until you pass under ref ht, or pass border
    while idx <= borderIdx and y[idx] > refHeight:
        idx = idx + 1
    
    return idx

# Returns xc, the x-coordinate center between xa and yb based on linear interpolation
def linterp(xa,xb,ya,yb,yc,bc):
    # interpolate btwn points (xa,ya) and (yb,yc) to find (xc,0.5*(yc-bc))
    xc = xa + (xb-xa) * (0.5*(yc+bc)-ya) / (yb-ya)
    
    # invoke LHopital if -Inf encountered
    if np.isnan(xc):
        # yc,yb guaranteed to be finite.
        if np.isinf(bc):
            # both ya and bc are -Inf
            xc = 0.5*(xa+xb)
        else:
            # only ya is -Inf
            xc = xb
    return xc

