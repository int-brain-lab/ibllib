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
# Ibllib tutorial example page
import examples.one.tutorial_script

# Ibllib ONE examples
import examples.one.behavior.number_mice_inproject
import examples.one.behavior.plot_weight_curve
import examples.one.behavior.plot_microphone_spectrogram
import examples.one.behavior.print_water_administrations

# Brainbox examples
# from brainbox.examples.titi import toto

# # ----- PRINT EXAMPLES -----
# # One examples
# print(examples.one.behavior.number_mice_inproject)
# print(examples.one.behavior.plot_weight_curve)
# print(examples.one.behavior.plot_microphone_spectrogram)
# print(examples.one.behavior.print_water_administrations)
#
# # Tutorial example page
# print(examples.one.tutorial_script)
#
# # Brainbox examples
# # print()

# ------ COMPUTE RUN TIME ----
end = perf_counter()
execution_time = (end - start)
print(execution_time)
