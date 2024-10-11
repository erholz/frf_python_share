import pickle

def get_TimeInfo(picklefile_dir):
    with open(picklefile_dir+'timeinfo.pickle', 'rb') as file:
        tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = pickle.load(file)
    return tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration

def get_FileInfo(picklefile_dir):
    with open(picklefile_dir+'fileinfo.pickle', 'rb') as file:
        local_base, lidarfloc, lidarext, noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext = pickle.load(file)
    return local_base, lidarfloc, lidarext, noaawlfloc, noaawlext, lidarhydrofloc, lidarhydroext

