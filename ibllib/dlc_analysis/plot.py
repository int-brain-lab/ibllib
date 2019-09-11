import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def plot_trajectory_colored(
    x,
    y,
    likelihood,
    likelihood_threshold=0,
    img=None,
    title="",
    OUTDIR="",
    FIGURE_STORE=False,
    cmap="jet",
):
    """
    Plot markers using x, y coordinates over image - if available
    The likelihood is used to color individual markers

    Inputs:
    ______
    x: array (D, T)
        The x coordinates of marker(s) in any given image
      D are the data dimensions (number of markers of a body part)
      T corresponds to time frames.
    y: array (D, T)
        The y coordinates of marker(s) in any given image
      D are the data dimensions (number of markers of a body part)
      T corresponds to time frames.
    likelihood_threshold: array (D, T)
      The likelihood for a given pair of (x,y) coordinates
      D are the data dimensions (number of markers of a body part)
      T corresponds to time frames.
    img: None or figure_class from matplotlib
        Background figure for markers to be plotted
      If None background is white image else uses figure given
    title: string
      Title for plot
    OUTDIR: string
      Location where to store the figure as .pdf
    FIGURE_STORE: boolean
        Flag to store plot
    cmap: color name from matplotlib color names
      Colormap to use to plot markers according to their likelihood values

    Outputs:
    _______
        NA
    """
    import matplotlib.pyplot as plt

    D, T = x.shape
    mask = likelihood >= likelihood_threshold

    for d in range(D):
        fig, ax = plt.subplots(figsize=(10, 8))
        if not (img is None):
            yn, xn, _ = img.shape
            ax.imshow(img)
            ax.set_xlim([0, xn])
            ax.set_ylim([yn, 0])

        plt.scatter(
            x[d, mask[d]],
            y[d, mask[d]],
            c=likelihood[d, mask[d]],
            marker="o",
            alpha=0.5,
            vmax=1.0,
            vmin=0,
            cmap=cmap,
        )

        plt.title(title)
        plt.xlabel("x coordinates in Arena")
        plt.ylabel("y coordinates in Arena")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        if FIGURE_STORE:
            fig.savefig(os.path.join(OUTDIR, title.replace(" ", "_") + ".pdf"))
        else:
            plt.show()
        return


def visualize_frames(clip, dxs, dys, t0=0, t0_len=1, exact_frames=None):
    """
    Visualize frames in clip, and superimpose markers given by dxs and dxy.

    :param clip: VideoClip class from moviepy
        See moviepy VideoClip documentation for details
    :param dxs: (D, T) array
    :param dys: (D, T) array
    :param t0: int
        index of minimum frame to plot
    :param t0_len: int
        length of frames to plot
    :param exact_frames: None or list
        if list, plot frames in exact_frames.
            if t0_len > 1,  selects subset of frames to plot from exact_frames
            else plots all frames in exact_frames
        if None:
            plots frames in range(t0, t0 + t0_len)
    :return:
    """
    xlim, ylim = clip.size
    fps = clip.fps
    if np.ndim(dxs) == 1:
        num_markers = 1
        dxs = dxs[None, :]
        dys = dys[None, :]
    else:
        num_markers = dxs.shape[0]

    color_class = plt.cm.ScalarMappable(cmap="cool")
    colors = color_class.to_rgba(np.linspace(0, 1, num_markers))

    if exact_frames is None:
        exact_frames = range(t0, t0 + t0_len)
    else:
        if t0_len > 1:
            exact_frames = np.random.choice(exact_frames, t0_len)

    for frame_idx in exact_frames:
        print(frame_idx)
        frame_idx_sec = frame_idx / fps

        title_ = "Frame id {} @ time {:.2f} [sec]".format(frame_idx, frame_idx_sec)
        plt.figure(figsize=(10, 8))
        plt.imshow(clip.get_frame(frame_idx_sec))

        for part_idx in range(num_markers):
            title_ += "\n ({:.2f}, {:.2f})".format(
                dxs[part_idx, frame_idx], dys[part_idx, frame_idx]
            )
            plt.plot(
                dxs[part_idx, frame_idx],
                dys[part_idx, frame_idx],
                c=colors[part_idx],
                marker="o",
                ms=10,
            )
            plt.title(title_)
        plt.tight_layout()
        plt.xlim([0, xlim])
        plt.ylim([ylim, 0])
        plt.show()
    return


def plot_masked_traces(
    xr,
    maskr=None,
    slices=[],
    title="Traces",
    trace_label="x",
    xlim=[],
    scaling_factor=1.05,
    figsize=(25, 15),
    plot_outlier=True,
    FIGURE_STORE=False,
    OUTDIR="",
):
    """
    Plot traces masked according to maskr
    :param xr: (D, T) array
        array to be plotted
    :param maskr (D, T) array
        part of traces not to plot
    :param slices: list of slice objects
        if given, adds slices to plot in black
        (used for outliers)
    :param title: string
        title of figure
    :param trace_label: string
        label of elements in x
    :param xlim: [xmin, xmax]
        tuple to limit extent of figure
    :param scaling_factor: float
        factor to scale each trace, to make them
        appear farther or closer together
    :param figsize: tuple
        dimensions of plot
    :param plot_outlier: bool
        flag to plot outliers
    :param FIGURE_STORE: bool
        flag to store plot
    :param OUTDIR: string
        directory where to store plot
    :return:
    """
    if np.ndim(xr) == 1:
        xr = xr[None, :]

    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    D, T = xr.shape

    if D > T:
        print("transpose array")
        xr = xr.T
        D, T = xr.shape

    if maskr is None:
        maskr = ~np.isnan(xr)

    tfull = np.arange(T)
    colors_ = sns.color_palette("husl", D)

    lim = scaling_factor * np.nanmax(abs(xr))

    for idx_ in range(0, D):
        seg = xr[idx_, :]

        # plot entire data
        ax.plot(
            tfull[maskr[idx_, :]],
            seg[maskr[idx_, :]] + lim * idx_,
            c=colors_[idx_],
            ls=" ",
            marker=".",
            label=r"${}_{}$ raw".format(trace_label, idx_ + 1),
        )

        # plot masked segments as squares
        if plot_outlier:
            ax.plot(
                tfull[~maskr[idx_, :]],
                seg[~maskr[idx_, :]] + lim * idx_,
                c=colors_[idx_],
                ls=" ",
                marker="s",
                markersize=8,
                label=r"${}_{}$ outlier".format(trace_label, idx_ + 1),
            )

        # PLot slices in traces
        if np.any(slices):
            for slice in slices:
                tslice = np.arange(slice.start, slice.stop)
                ax.plot(tslice, xr[idx_, slice] + lim * idx_, "k")

    ax.legend(loc="best", bbox_to_anchor=(1, 1))
    # plt.yticks(lim * np.arange(D), ["$x_{}$".format(d + 1) for d in range(D)])
    # plt.xlim([1500,3000])
    if np.any(xlim):
        plt.xlim(xlim)
    else:
        plt.xlim([0, T])

    plt.tight_layout()
    if FIGURE_STORE:
        fig.savefig(os.path.join(OUTDIR, title.replace(" ", "_") + ".pdf"))
    else:
        plt.show()
    return


def plot_group_slices_lens(
    slices_groups, OUTDIR="", FIGURE_STORE=False, fname="dist_cont_length"
):
    """
    Make violin plot given slices_groups.
    :param slices_groups: dict
        eack key is an index and the values are the lengths of the
        segments for that given key
    :param OUTDIR: string
        directory where to store plot
    :param FIGURE_STORE: bool
        flag to store plot
    :param fname: string
        name of file
    :return:
    """
    df = pd.DataFrame.from_dict(slices_groups, orient="index").transpose()
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    sns.violinplot(data=df, split=True, scale="count", ax=ax)
    ax.set_xlabel("Marker Groups")
    ax.set_ylabel("Continuous Segment Lengths")
    ax.set_title("Distribution of continuous segment lengths")

    plt.tight_layout()
    if FIGURE_STORE:
        fig.savefig(os.path.join(OUTDIR, fname + ".pdf"))
    else:
        plt.show()
    plt.close()
    return


#%%
