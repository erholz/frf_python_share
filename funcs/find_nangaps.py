import numpy as np

def find_nangaps(y_input):
    """

       :param y_input: consists of non-singular array
       :return: gapstart, gapend, gapsize, maxgap
       """

    if sum(np.isnan(y_input)) == 0:
        gapstart = np.nan
        gapend = np.nan
        gapsize = np.array([0])
        maxgap = np.array([0])
    elif sum(np.isnan(y_input)) == 1:
        gapstart = np.where(np.isnan(y_input))
        gapend = np.where(np.isnan(y_input))
        gapsize = np.array([1])
        maxgap = np.array([1])
    else:
        numcumnan = np.empty(shape=y_input.shape)
        numcumnan[:] = np.nan
        tmp = np.cumsum(np.isnan(y_input), axis=0)
        numcumnan[tmp > 0] = tmp[tmp > 0]
        uniq_numcumnan = np.unique(numcumnan)
        uniq_numcumnan = uniq_numcumnan[~np.isnan(uniq_numcumnan)]
        tmpgapstart = []
        tmpgapend = []
        for ij in np.arange(uniq_numcumnan.size):
            # If there is only ONE entry of a cumnan value, then we know it's a new nan value OR it's the end of the vector
            if sum((numcumnan == uniq_numcumnan[ij])) == 1:
                tmp = np.where(numcumnan == uniq_numcumnan[ij])[0]
                tmpgapstart = np.append(tmpgapstart,tmp[0])
                # if tmp is the END of the vector, also designate as tmpgapend
                if tmp == len(y_input)-1:
                    tmpgapend = np.append(tmpgapend,tmp[0])
            # If there are multiple entries of a cumnan value, then we know it switches from nan to not-nan
            elif sum((numcumnan == uniq_numcumnan[ij])) > 1:
                tmp = np.where(numcumnan == uniq_numcumnan[ij])[0]
                # the first value of tmp is where it switches from nan to not-nan, the last would be the first before the next nan (if it exists)
                tmpgapend = np.append(tmpgapend,tmp[0])
                # if there is more than 1 instance of cumnan but no preceding nan, then it is ALSO a starting nan
                if ~np.isnan(y_input[tmp[0]-1]):
                    tmpgapstart = np.append(tmpgapstart,tmp[0])
                # if it is the FIRST value, then it is ALSO a tmpgapstart
                if tmp[0] == 0:
                    tmpgapstart = np.append(tmpgapstart, tmp[0])
        # new revisions may create duplicates....
        tmpgapend = np.unique(tmpgapend)
        tmpgapstart = np.unique(tmpgapstart)
        gapend = tmpgapend[:]
        # if NO tmpgapstart have been found, then we have multiple single-nans
        if len(tmpgapstart) == 0:
            gapstart = gapend[:]
        else:
            gapstart = [tmpgapstart[0]]
            if len(tmpgapstart) > 0:
                # now, we need to figure out if this is in the beginning of the gap or the middle
                tmp1 = np.diff(tmpgapstart)     # tmp1 is GRADIENT in tmpgapstart (diff of 1 indicates middle of gap)
                tmp2 = tmpgapstart[1:]          # tmp2 is all the tmpgapstarts OTHER than the first
                # if there is only ONE gap of 3 nans, there will be no tmp1 not equal to 1...
                if sum(tmp1 != 1) > 0:
                    # tmpid = where numcumnan is equal to a gap that is not in the middle (tmp1 != 1)
                    tmpid = tmp2[tmp1 != 1]
                    gapstart = np.append(gapstart, tmpid)
            if len(gapend) > len(gapstart):
                for ij in np.arange(gapend.size):
                    # if there is a gapend that is surrounded by non-nans, then it is a single-nan gap
                    if gapend[ij] == len(y_input)-1:
                        if ~np.isnan(y_input[int(gapend[ij]) - 1]):
                            missinggapstart = gapend[ij]
                            gapstart = np.append(missinggapstart, gapstart)
                    else:
                        if ~np.isnan(y_input[int(gapend[ij])-1]) & ~np.isnan(y_input[int(gapend[ij])+1]):
                            missinggapstart = gapend[ij]
                            gapstart = np.append(missinggapstart, gapstart)
            if np.max(gapstart) > np.max(gapend):
                gapend = np.append(gapend, np.max(gapstart))
        gapend = np.unique(gapend)
        gapstart = np.unique(gapstart)
        gapend = np.array(sorted(gapend))
        gapstart = np.array(sorted(gapstart))
        gapsize = (gapend - gapstart) + 1
        maxgap = np.nanmax(gapsize)

    return gapstart, gapend, gapsize, maxgap
