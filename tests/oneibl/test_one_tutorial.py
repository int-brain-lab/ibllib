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
# Tutorial example page
import examples.one.tutorial_script

# One examples
from examples.one.behavior import plot_weight_curve

# Brainbox examples
# from brainbox.examples.titi import toto

# ----- PRINT EXAMPLES -----
# One examples
print(plot_weight_curve)

# Tutorial example page
print(examples.one.tutorial_script)

# Brainbox examples
# print()

# ------ COMPUTE RUN TIME ----

end = perf_counter()
execution_time = (end - start)
print(execution_time)
