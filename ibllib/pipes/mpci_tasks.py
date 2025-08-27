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
import re
import matplotlib.pyplot as plt
import sparse

import logging
_logger = logging.getLogger(__name__)

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
        Loads the suite2p ops file to retrieve the dimensions of the data.bin file. This is now lazily loaded from a
        zip file

        Returns
        -------
        (int, int, int)
            number of frames, number of y pixels, number of x pixels.
        """
        s2p_ops = np.load(self.ops_path, allow_pickle = True)['ops'].item()
        return s2p_ops['nframes'], s2p_ops['Ly'], s2p_ops['Lx']

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]):
        return self.data[item].copy()

class PMD(base_tasks.MesoscopeTask, base_tasks.RegisterRawDataTask):

    gpu=1
    env='masknmftoolbox'
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
            'train_network': True,
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

            'output_files': [('_ibl_mpciU.spatial_basis.sparse_npz', 'alf/FOV*', True),
                             ('_ibl_mpciU.projector.sparse_npz', 'alf/FOV*', True),
                             ('_ibl_mpciV.temporal_basis', 'alf/FOV*', True),
                             ('_ibl_mpciU.meanImage', 'alf/FOV*', True),
                             ('_ibl_mpciU.stdImage', 'alf/FOV*', True)]  # False = not required output
        }
        return signature

    def _load_bin_file(self,
                       s2p_folderpath: Union[str, bytes, os.PathLike],
                       alf_folderpath: Union[str, bytes, os.PathLike]) -> np.ndarray:
        bin_path = os.path.join(s2p_folderpath, "imaging.frames_motionRegistered.bin")
        ops_path = os.path.join(alf_folderpath, '_suite2p_ROIData.raw.zip')
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
            logging.warning("Running PMD to generate training data on CPU, this will very slow")

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
            logging.warning("Running compression on CPU")

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

        logging.info(f"processing complete. denoiser rank is {pmd_output.pmd_rank}")
        return pmd_output


    def _qc_results(self,
                    my_data: np.ndarray,
                    pmd_array: masknmf.PMDArray,
                    snapshot_folder: pathlib.Path,
                    fov_identifier:str,
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

        lag1_image_path = snapshot_folder / Path(f"lag1_img_{fov_identifier}.png")
        spatial_image_path = snapshot_folder / Path(f"spatial_img_{fov_identifier}.png")

        self.plot_summary_images_side_by_side([raw_lag1, pmd_lag1, resid_lag1],
                                              ['Raw Lag1 Autocov / Raw Std',
                                                    'PMD Lag1 Autocov / Raw Std',
                                                    'Resid Lag1 Autocov / Raw Std'],
                                                    filename = lag1_image_path)

        self.plot_summary_images_side_by_side([raw_spatial_corr, pmd_spatial_corr, residual_spatial_corr],
                                              ['Raw Spatial Corr / Raw Std',
                                               'PMD Spatial Corr / Raw Std',
                                               'Resid Spatial Corr / Raw Std'],
                                              filename = spatial_image_path)

    def plot_summary_images_side_by_side(self,
                                         images,
                                         titles,
                                         filename='covariances.png',
                                         cmap='viridis'):
        """
        Plots 3 covariance images side by side and saves the figure as a PNG.

        Parameters
        ----------
        images: list of 3 images, each np.ndarray
            2D arrays representing covariance images.
        titles: list of 3 strings, one title for each image
        filename : str
            The absolute output path where we save the file.
        cmap : str
            Colormap used to display the images (default is 'viridis').
        """
        cov1, cov2, cov3 = images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, cov, title in zip(axes, [cov1, cov2, cov3], titles):
            im = ax.imshow(cov, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    def _run_singlevideo(self,
                         s2p_folderpath: pathlib.Path,
                         alf_folderpath: pathlib.Path,
                         snapshot_path: pathlib.Path,
                         fov_identifier: str,
                         **kwargs):
        """
        Run the PMD processing task on a single plane of data (1 video)

        Parameters
        ----------
        s2p_folderpath: A path of the form: '/path/to/subject/YYYY-MM-DD/XXX/suite2p/plane{i}'. This folder contains
            the motion corrected .bin files
        alf_folderpath: A path of the form '/path/to/subject/YYYY-MM-DD/XXX/alf/FOV{i}'. This folder contains the ops file
            in a .zip
        kwargs
            Optional parameters that may be used in the future.
        
        Returns
        -------
        file_paths : list of pathlib.Path
            List of file paths that were processed or generated by this task.
        """
        # Merge default configs with user-provided configs from _run
        logging.info("Merging Configs")
        cfg = OmegaConf.merge(self.cfg, kwargs)

        logging.info("Updating GPU status")
        self.gpu = 1 if cfg.device.startswith("cuda") else 0

        logging.info("Loading bin file")
        # Load the .bin file
        my_data= self._load_bin_file(s2p_folderpath, alf_folderpath)

        if cfg.train_network:
            logging.info("Training neural net")
            #Train the neural network temporal denoiser
            denoiser = self._train_denoiser(my_data, cfg)
        else:
            logging.info("No neural net used for this data")
            denoiser = None

        logging.info("Compressing data")
        #Use the neural network temporal denoiser to compress the data
        pmd_array = self._compress_data(my_data, cfg, denoiser=denoiser)

        logging.info("Making QC metrics")
        #Generate some quality control metrics
        self._qc_results(my_data,
                         pmd_array,
                         snapshot_path,
                         fov_identifier,
                         device=cfg.device,
                         frame_batch_size=cfg.frame_batch_size)

        logging.info(f"Saved snapshots for fov {fov_identifier}")
        logging.info(f"Saving the PMD results in the ONE format")
        u_gxcs = self._sparse_u_to_gxcs(pmd_array.u,
                                        pmd_array.shape[1],
                                        pmd_array.shape[2])
        u_projector_gxcs = self._sparse_u_to_gxcs(pmd_array.u_local_projector,
                                                  pmd_array.shape[1],
                                                  pmd_array.shape[2])
        v = pmd_array.v.cpu().numpy()
        mean_img = pmd_array.mean_img.cpu().numpy()
        var_img = pmd_array.var_img.cpu().numpy()

        output_file_list = []
        curr_path = alf_folderpath.joinpath('_ibl_mpciU.spatial_basis.sparse_npz')
        with open(curr_path, 'wb') as fp:
            sparse.save_npz(fp, u_gxcs)
        output_file_list.append(curr_path)

        curr_path = alf_folderpath.joinpath('_ibl_mpciU.projector.sparse_npz')
        with open(curr_path, 'wb') as fp:
            sparse.save_npz(fp, u_projector_gxcs)
        output_file_list.append(curr_path)

        curr_path = alf_folderpath.joinpath('_ibl_mpciV.temporal_basis.npy')
        np.save(curr_path, v)
        output_file_list.append(curr_path)

        curr_path = alf_folderpath.joinpath('_ibl_mpciU.meanImage.npy')
        np.save(curr_path, mean_img)
        output_file_list.append(curr_path)

        curr_path = alf_folderpath.joinpath('_ibl_mpciU.stdImage.npy')
        np.save(curr_path, var_img)
        output_file_list.append(curr_path)

        return output_file_list

    def _sparse_u_to_gxcs(self,
                          u: torch.sparse_coo_tensor,
                          fov_dim1: int,
                          fov_dim2: int):
        """
        Given a sparse u matrix of shape (fov_dim1*fov_dim2, pmd_rank), we want to construct a gxcs sparse object
        of shape (fov_dim1, fov_dim2, pmd_rank). The assumption is that u has columns which are vectorized in row
        major order (so reshaping from (fov_dim1*fov_dim2,) --> (fov_dim1, fov_dim2) involves using shape C).
        """
        row_indices, col_indices = u.indices()
        pmd_rank = u.shape[1]

        # Convert row indices back to (height, width)
        height_indices = (row_indices // fov_dim2).cpu().numpy()
        width_indices = (row_indices % fov_dim2).cpu().numpy()
        col_indices = col_indices.cpu().numpy()
        values = u.values().cpu().numpy()

        # Stack indices as (ndim, nnz)
        final_ind = np.vstack([height_indices, width_indices, col_indices])
        s = sparse.COO(final_ind, values, shape=(fov_dim1, fov_dim2, pmd_rank))
        return sparse.GCXS.from_coo(s)


    def _generate_per_dataset_input_paths(self):
        """
        Returns a list of tuples, describing relevant paths for each plane being processed. Each tuple has 2 paths:
        (1) A path to the suite2p plane folder that is being processed
        (2) A path to the alf FOV describing this plane
        """
        # Identify the plane paths
        s2p_path = self.session_path.joinpath('suite2p')
        pattern = re.compile(r'(?<=^plane)\d+$')
        s2p_plane_paths = sorted(s2p_path.glob('plane?*'), key=lambda x: int(pattern.search(x.name).group()))

        # Identify the alf output paths corresponding to each plane
        alf_fov_paths = [f"FOV_{int(plane.stem.replace('plane', '')):02d}" for plane in s2p_plane_paths]
        alf_fov_paths = [self.session_path.joinpath('alf/').joinpath(i) for i in alf_fov_paths]

        return zip(s2p_plane_paths, alf_fov_paths)


    def _run(self, **kwargs):
        # Make a folder where snapshots are stored:
        snapshot_path = self.session_path / Path("pmd_qc_snapshots")
        snapshot_path.mkdir(parents=True, exist_ok=True)
        #Process each dataset
        file_data = self._generate_per_dataset_input_paths()
        output_file_list = []
        for i, elt in enumerate(list(file_data)):
            print(f"Processing fov {i}")
            s2p_folderpath, alf_folderpath = elt
            fov_identifier = alf_folderpath.stem
            curr_file_list = self._run_singlevideo(s2p_folderpath,
                                  alf_folderpath,
                                  snapshot_path,
                                  fov_identifier,
                                  **kwargs)
            output_file_list.extend(curr_file_list)


        ## COMMENTING THIS OUT UNTIL CLEARER HOW TO USE THIS
        #self.register_snapshots(unlink=False, collection=snapshot_path)

        return output_file_list


import unittest

class TestPMD(unittest.TestCase):

    def setUp(self):
        # self.session_path = '/path/to/subject/YYYY-MM-DD/XXX'
        self.session_path = '/media/app2139/SanDisk_1/IBL_Alyx/cortexlab/Subjects/SP067/2025-06-03/001/'
        self.pmd_task = PMD(self.session_path)

    def test_pmd_processing(self):
        self.pmd_task._run(device='cuda',
                           train_network=False)


if __name__ == '__main__':
    unittest.main()