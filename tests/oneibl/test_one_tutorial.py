"""
Import examples so as to run them as test.
Printing is used to request the import.
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


# - Histology
import examples.one.histology.register_lasagna_tracks_alyx

# -- Brainbox examples
# from brainbox.examples.titi import toto

# ------ COMPUTE RUN TIME ----
end = perf_counter()
execution_time = (end - start)
print(execution_time)
