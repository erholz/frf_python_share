import pickle

def get_TimeInfo():
    with open('timeinfo.pickle', 'rb') as file:
        tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration = pickle.load(file)
    return tzinfo, time_format, time_beg, time_end, epoch_beg, epoch_end, TOI_duration
