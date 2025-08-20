from ibllib.pipes import base_tasks
from ibllib.oneibl.data_handlers import ExpectedDataset
from typing import *
import os
import pathlib
from pathlib import Path
import numpy as np
import masknmf
from omegaconf import DictConfig, OmegaConf
import hydra
from masknmf import display
import torch

class MotionBinDataset:
    """Load a suite2p data.bin imaging registration file."""

    def __init__(self,
                 data_path: Union[str, pathlib.Path],
                 metadata_path: Union[str, pathlib.Path]):
        """
        Load a suite2p data.bin imaging registration file.

        Parameters
        ----------
        data_path (str, pathlib.Path): The session path containing preprocessed data.
        metadata_path (str, pathlib.Path): The metadata_path to load.
        """
        self.bin_path = Path(data_path)
        self.ops_path = Path(metadata_path)
        self._dtype = np.int16
        self._shape = self._compute_shape()
        self.data = np.memmap(self.bin_path, mode='r', dtype=self.dtype, shape=self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self):
        """
        This property should return the shape of the dataset, in the form: (d1, d2, T) where d1
        and d2 are the field of view dimensions and T is the number of frames.

        Returns
        -------
        (int, int, int)
            The number of y pixels, number of x pixels, number of frames.
        """
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def _compute_shape(self):
        """
        Loads the suite2p ops file to retrieve the dimensions of the data.bin file.

        Returns
        -------
        (int, int, int)
            number of frames, number of y pixels, number of x pixels.
        """
        ops_file = self.ops_path
        if ops_file.exists():
            ops = np.load(ops_file, allow_pickle=True).item()
        return ops['nframes'], ops['Ly'], ops['Lx']

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]):
        return self.data[item].copy()

class PMD(base_tasks.MesoscopeTask):

    def __init__(self,
                 session_path: Union[str, bytes, os.PathLike],
                 **kwargs):
        super().__init__(session_path=session_path, **kwargs)
        self._initialize_params()

    def _initialize_params(self):
        config_dict = {
            'path': '/path/to/data/',
            'outdir': '.',
            'block_size_dim1': 32,
            'block_size_dim2': 32,
            'background_rank': 0,
            'max_components': 20,
            'max_consecutive_failures': 1,
            'spatial_avg_factor': 1,
            'temporal_avg_factor': 1,
            'device': 'cpu',
            'frame_batch_size': 1024,
            'num_pixels_ignore': 5, # How many pixels on the borders we ignore when training
            ## For training the network:
            'epochs': 5,
            'learning_rate': 1e-4,
        }
        self.cfg = OmegaConf.create(config_dict)
    @property
    def signature(self):
        # The number of in and outputs will be dependent on the number of input raw imaging folders and output FOVs
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I('_suite2p_ROIData.raw.zip', 'alf/FOV*', True, unique=False),
                            I('imaging.frames_motionRegistered.bin', 'suite2p/plane*', True, unique=False)],
            # FIXME come up with dataset names for outputs
            'output_files': [('imaging.sparse1.sparse_npz', 'alf/FOV*', True),
                             ('imaging.sparse2.sparse_npz', 'alf/FOV*', True),
                             ('imaging.dense.npz', 'alf/FOV*', True),
                             ('model.weights.npy', 'alf/FOV*', False)]  # False = not required output
        }
        return signature

    def _load_bin_file(self,
                       session_path: Union[str, bytes, os.PathLike]) -> np.ndarray:
        bin_path = os.path.join(session_path, "data.bin")
        ops_path = os.path.join(session_path, "ops.npy")
        my_data = MotionBinDataset(bin_path, ops_path)[:]
        return my_data

    def _train_denoiser(self,
                        my_data: np.ndarray,
                        cfg: DictConfig) -> torch.nn.Module:

        num_frames, fov_dim1, fov_dim2 = my_data.shape

        if cfg.num_pixels_ignore * 2 >= fov_dim1 or cfg.num_pixels_ignore * 2 >= fov_dim2:
            raise ValueError("Number of pixels to ignore at the borders is too large")

        mask = np.zeros((my_data.shape[1], my_data.shape[2]))
        mask[cfg.num_pixels_ignore:-1 * cfg.num_pixels_ignore, cfg.num_pixels_ignore:-1 * cfg.num_pixels_ignore] = 1.0

        block_sizes = [cfg.block_size_dim1, cfg.block_size_dim2]

        device = cfg.device
        if device == 'cpu':
            display("Running PMD to generate training data on CPU")

        pmd_obj = masknmf.compression.pmd_decomposition(my_data,
                                                        block_sizes,
                                                        my_data.shape[0],
                                                        max_components=cfg.max_components,
                                                        max_consecutive_failures=cfg.max_consecutive_failures,
                                                        temporal_avg_factor=cfg.temporal_avg_factor,
                                                        spatial_avg_factor=cfg.spatial_avg_factor,
                                                        background_rank=cfg.background_rank,
                                                        device=device,
                                                        frame_batch_size=cfg.frame_batch_size)

        v = pmd_obj.v.cpu()
        trained_model, _ = masknmf.compression.denoising.train_total_variance_denoiser(v,
                                                                                       max_epochs=5,
                                                                                       batch_size=128,
                                                                                       learning_rate=1e-4)

        trained_nn_module = masknmf.compression.PMDTemporalDenoiser(trained_model)
        return trained_nn_module

    def _compress_data(self,
                       my_data: np.ndarray,
                       cfg: DictConfig,
                       denoiser: Optional[torch.nn.Module] = None) -> masknmf.PMDArray:
        """
        Runs PMD compression/denoising, with or without a neural network denoising module

        Parameters
        ----------
        my_data
        """
        # Load the data

        device = cfg.device
        if device == 'cpu':
            display("Running compression on CPU")

        if denoiser is not None:
            pmd_output = masknmf.compression.pmd_decomposition(my_data,
                                                               (cfg.block_size_dim1, cfg.block_size_dim2),
                                                               my_data.shape[0],
                                                               max_components=cfg.max_components,
                                                               max_consecutive_failures=cfg.max_consecutive_failures,
                                                               temporal_avg_factor=cfg.temporal_avg_factor,
                                                               spatial_avg_factor=cfg.spatial_avg_factor,
                                                               background_rank=cfg.background_rank,
                                                               device=device,
                                                               temporal_denoiser=denoiser,
                                                               frame_batch_size=cfg.frame_batch_size)
        else:
            pmd_output = masknmf.compression.pmd_decomposition(my_data,
                                                               (cfg.block_size_dim1, cfg.block_size_dim2),
                                                               my_data.shape[0],
                                                               max_components=cfg.max_components,
                                                               max_consecutive_failures=cfg.max_consecutive_failures,
                                                               temporal_avg_factor=cfg.temporal_avg_factor,
                                                               spatial_avg_factor=cfg.spatial_avg_factor,
                                                               background_rank=cfg.background_rank,
                                                               device=device,
                                                               temporal_denoiser=None,  # Turn off denoiser
                                                               frame_batch_size=cfg.frame_batch_size)

        display(f"processing complete. denoiser rank is {pmd_output.pmd_rank}")
        return pmd_output

    def _qc_results(self,
                    my_data: np.ndarray,
                    pmd_array: masknmf.PMDArray,
                    device: str = "cpu",
                    frame_batch_size: int = 100):
        raw_spatial_corr, pmd_spatial_corr, residual_spatial_corr = masknmf.diagnostics.compute_pmd_spatial_correlation_maps(
            my_data,
            pmd_array,
            device=device,
            batch_size=frame_batch_size)

        raw_lag1, pmd_lag1, resid_lag1 = masknmf.diagnostics.pmd_autocovariance_diagnostics(my_data,
                                                                                            pmd_array,
                                                                                            device=device,
                                                                                            batch_size=frame_batch_size)

        raw_spatial_corr = raw_spatial_corr.cpu().numpy()
        pmd_spatial_corr = pmd_spatial_corr.cpu().numpy()
        residual_spatial_corr = residual_spatial_corr.cpu().numpy()

        raw_lag1 = raw_lag1.cpu().numpy()
        pmd_lag1 = pmd_lag1.cpu().numpy()
        resid_lag1 = resid_lag1.cpu().numpy()
        return raw_spatial_corr, pmd_spatial_corr, residual_spatial_corr, raw_lag1, pmd_lag1, resid_lag1

    def _run(self, **kwargs):
        """
        Run the PMD processing task.

        Parameters
        ----------
        kwargs
            Optional parameters that may be used in the future.
        
        Returns
        -------
        file_paths : list of pathlib.Path
            List of file paths that were processed or generated by this task.
        """
        # Merge default configs with user-provided configs from _run
        display("Merging Configs")
        cfg = OmegaConf.merge(self.cfg, kwargs)

        display("Loading bin file")
        # Load the .bin file
        my_data= self._load_bin_file(self.session_path)

        display("Training neural net")
        #Train the neural network temporal denoiser
        denoiser = self._train_denoiser(my_data, cfg)

        display("Compressing data")
        #Use the neural network temporal denoiser to compress the data
        pmd_array = self._compress_data(my_data, cfg, denoiser=denoiser)

        display("Making QC metrics")
        #Generate some quality control metrics
        outputs = self._qc_results(my_data,
                                   pmd_array,
                                   device=cfg.device,
                                   frame_batch_size=cfg.frame_batch_size)

        ##TODO: Return file paths

import unittest

class TestPMD(unittest.TestCase):

    def setUp(self):
        # self.session_path = '/path/to/subject/YYYY-MM-DD/XXX'
        self.session_path = '/media/app2139/SanDisk_1/IBL_Alyx/cortexlab/Subjects/SP067/2025-06-03/001/suite2p/plane1/'
        self.pmd_task = PMD(self.session_path)

    def test_pmd_processing(self):
        self.pmd_task._run(device='cuda')


if __name__ == '__main__':
    unittest.main()