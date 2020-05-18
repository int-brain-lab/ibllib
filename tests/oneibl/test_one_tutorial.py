"""
Import examples so as to run them as test (alphabetical order).
Examples have to be able to run upon adding.
"""
# Author: Olivier, Gaelle

# ----- IMPORT TIME MODULE -----
from time import perf_counter
start = perf_counter()

# ----- IMPORT EXAMPLES -----
# -- Ibllib tutorial example page
import examples.one.tutorial_script

# -- Ibllib ONE examples
# - Behavior
import examples.one.behavior.number_mice_inproject
import examples.one.behavior.plot_microphone_spectrogram
import examples.one.behavior.plot_weight_curve
import examples.one.behavior.print_water_administrations
import examples.one.behavior.water_administrations_add_new
import examples.one.behavior.water_administrations_weekend

# - Ephys
import examples.one.ephys.get_list_mice_certif
import examples.one.ephys.get_list_mice_repeated_site
import examples.one.ephys.get_probe_label_dir
import examples.one.ephys.get_spikeData_and_brainLocations
import examples.one.ephys.raw_data_decompress
import examples.one.ephys.raw_data_download
import examples.one.ephys.raw_data_sync_session_time

# - Histology
import examples.one.histology.get_probe_trajectory
import examples.one.histology.register_lasagna_tracks_alyx
import examples.one.histology.visualization3D_alyx_traj_planned_histology
# TODO import examples.one.histology.visualization3D_repeated_site
# TODO import examples.one.histology.visualization3D_rotating_gif_firstpassmap_plan
import examples.one.histology.visualization3D_rotating_gif_selectedmice  # TODO out path
import examples.one.histology.visualization3D_subject_channels  # TODO check docstring
import examples.one.histology.visualization3D_subject_histology
import examples.one.histology.visualize_alyx_channels_coronal  # TODO Delete?
import examples.one.histology.visualize_alyx_traj_coronal_sagittal_raster
import examples.one.histology.visualize_session_coronal_tilted

# TODO error in : import examples.one.histology.visualize_track_file_coronal_GUIoption
# TODO error in: import examples.one.histology.visualize_track_file_coronal_sagittal_slice
#  Incompatible library version:
#  libopencv_freetype.4.1.dylib requires version 24.0.0 or later, but libfreetype.6.dylib provides version 23.0.0


# -- Brainbox examples
# from brainbox.examples.titi import toto

# ------ COMPUTE RUN TIME ----
end = perf_counter()
execution_time = (end - start)
print(execution_time)
