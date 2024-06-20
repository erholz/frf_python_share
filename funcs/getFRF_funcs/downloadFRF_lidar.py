
# !/usr/bin/env python
# Script to download .nc files from a THREDDS catalog directory

from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve
import numpy as np
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from funcs.get_timeinfo import get_FileInfo

local_base, lidarfloc, lidarext, noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext = get_FileInfo()

# Divide the url you get from the data portal into two parts
# Everything before "catalog/"
server_url = 'https://chldata.erdc.dren.mil/thredds/'



# Everything after "catalog/"
# # For lidar transects
# request_url = 'catalog/frf/geomorphology/elevationTransects/duneLidarTransect/'
# local_dir = '/volumes/macDrive/lidarTransects/'
# years = np.arange(2015,2025)

# For lidar hydro
request_url = 'catalog/frf/oceanography/waterlevel/eopNoaaTide/'
local_dir = local_base + '/waterlevel/eopNOAA/'
# local_dir = lidarfloc
years = np.arange(2015,2025)

def get_elements(url, tag_name, attribute_name):
    """Get elements from an XML file"""
    # usock = urllib2.urlopen(url)
    usock = urlopen(url)
    xmldoc = minidom.parse(usock)
    usock.close()
    tags = xmldoc.getElementsByTagName(tag_name)
    attributes = []
    for tag in tags:
        attribute = tag.getAttribute(attribute_name)
        attributes.append(attribute)
    return attributes


def main():
    for year in years:
        url = server_url + request_url + str(year) + '/catalog.xml'
        print(url)
        catalog = get_elements(url, 'dataset', 'urlPath')
        files = []
        for citem in catalog:
            if (citem[-3:] == '.nc'):
                files.append(citem)
        count = 0

        file_subset = files#[0:12]

        for f in file_subset:
            count += 1
            file_url = server_url + 'fileServer/' + f
            file_prefix = file_url.split('/')[-1][:-3]
            file_name = file_prefix + '.nc'
            #file_name = file_prefix + '_' + str(count) + '.nc'

            print('Downloaing file %d of %d' % (count, len(file_subset)))
            print(file_url)
            print(file_name)
            if not os.path.isdir(local_dir):
                os.makedirs(local_dir)
            a = urlretrieve(file_url, local_dir + file_name)
            print(a)

    return catalog, files, file_subset

# Run main function when in comand line mode
if __name__ == '__main__':
    catalog, files, file_subset = main()