
# !/usr/bin/env python
# Script to download .nc files from a THREDDS catalog directory

from xml.dom import minidom
from urllib.request import urlopen
from urllib.request import urlretrieve
import numpy as np

# Divide the url you get from the data portal into two parts
# Everything before "catalog/"
server_url = 'https://chldata.erdc.dren.mil/thredds/'

# Everything after "catalog/"
# # For lidar transects
# request_url = 'catalog/frf/geomorphology/elevationTransects/duneLidarTransect/'
# local_dir = '/volumes/macDrive/lidarTransects/'
# years = np.arange(2015,2025)

# # For lidar hydro
# request_url = 'catalog/frf/oceanography/waves/lidarHydrodynamics/'
# local_dir = '/volumes/macDrive/FRF_Data/waves_lidar/lidar_hydro/'
# years = np.arange(2023,2025)

# # For waverider-26m
# request_url = 'catalog/frf/oceanography/waves/waverider-26m/'
# local_dir = '/volumes/macDrive/FRF_Data/waverider-26m/'
# years = np.arange(2008,2025)

# For waverider-26m
request_url = 'catalog/wis/Pacific/ST84065/'
local_dir = '/volumes/anderson/WIS84065/'
years = np.arange(1980,2024)

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
            a = urlretrieve(file_url, local_dir + file_name)
            print(a)

    return catalog, files, file_subset

# Run main function when in comand line mode
if __name__ == '__main__':
    catalog, files, file_subset = main()