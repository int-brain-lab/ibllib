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
import examples.one.ephys.get_LFP_decompress
import examples.one.ephys.get_list_mice_certif
import examples.one.ephys.get_list_mice_repeated_site
import examples.one.ephys.get_probe_label_dir
import examples.one.ephys.get_spikeData_and_brainLocations

# - Histology
import examples.one.histology.register_lasagna_tracks_alyx

# -- Brainbox examples
# from brainbox.examples.titi import toto

# ------ COMPUTE RUN TIME ----
end = perf_counter()
execution_time = (end - start)
print(execution_time)
