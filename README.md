# frf_python_share
Public repository to process FRF mounted LIDAR topo files for morphology calculations

To run, open run_code.py and modify the following:
  (1) local_base - name of main directory in which all downloaded FRF data is contained
  (2) Period of interest, including time_beg, time_end, and tzinfo
  (3) cont_elev - array of elevations at which contours are extracted
  (4) num_profs_plot - number of topo-bathy profiles (equally spaced in time) to plot to visualize period of interest
  (5) lidarfloc - the subdirectory of local_base containing all available LIDAR files
  (6) noaawlfloc - the subdirectory of local_base containing all available NOAA water level files
  (7) lidarhydrofloc - the subdirectory of local_base containing all available LIDAR Hydro files

