import os
import numpy as np
import datetime as dt
from bs4 import BeautifulSoup
import requests
import re

# prep file collection
def find_files_in_range(floc,ext,epoch_beg,epoch_end,tzinfo):
    list_of_files = np.array(find_files_local(floc, ext))
    fdate = []
    for fname_ii in list_of_files:
        fdate = np.append(fdate, int(fname_ii[-9:-3]))
    sorted_list = list_of_files[np.argsort(fdate)]
    sorted_fdate = fdate[np.argsort(fdate)]
    # Find names within the time of interest
    fname_epoch = []
    for datejj in sorted_fdate:
        datejj_str = str(datejj)
        tmp_epoch = dt.datetime.strptime(datejj_str[0:6], '%Y%m').timestamp()
        fname_epoch.append(tmp_epoch)
    fname_epoch = np.array(fname_epoch)
    ij_in_range = (fname_epoch >= epoch_beg) & (fname_epoch <= epoch_end)
    if sum(ij_in_range) == 0:
        # check if desired [year/month] has match in sorted_fdate
        begnum = int(dt.datetime.fromtimestamp(epoch_beg,tzinfo).strftime('%Y%m'))
        endnum = int(dt.datetime.fromtimestamp(epoch_end,tzinfo).strftime('%Y%m'))
        ij_in_range = (begnum == sorted_fdate ) | (endnum == sorted_fdate)
        if sum(ij_in_range) == 0:
            print('No data to see here, folks')
            print('Try another data range')
            #exit()
            return np.empty((0,0))
    if min(np.argwhere(ij_in_range)) > 0:
        ij_in_range[min(np.argwhere(ij_in_range))-1] = True
    if max(np.argwhere(ij_in_range)) < ij_in_range.size - 1:
        ij_in_range[max(np.argwhere(ij_in_range)) + 1] = True
    fname_in_range = sorted_list[ij_in_range]
    return fname_in_range

def find_files_thredds(floc,ext_in):
    frf_base = 'https://chlthredds.erdc.dren.mil/thredds/catalog/frf/'
    url_in = frf_base + floc
    soup = BeautifulSoup(requests.get(url_in).text, "html.parser")
    # soup = BeautifulSoup(requests.get(url).text, "html.parser")
    ids = []
    for tag in soup.find_all('dataset', id=re.compile(ext_in)):
        ids.append(tag['name'])
    return ids

def find_files_local(floc,ext_in):
    full_path = floc
    ids = []
    for file in os.listdir(full_path):
        if file.endswith(ext_in):
            if not file.startswith('.'):
                ids.append(file)
    return ids
