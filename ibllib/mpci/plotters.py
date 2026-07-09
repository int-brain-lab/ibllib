import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_triangle(triangle_points, axes=None, **kwargs):
    if axes is None:
        axes = plt.figure().add_subplot(projection="3d")

    if "color" not in kwargs:
        kwargs["color"] = "k"

    a, b, c = triangle_points
    points = np.concatenate(
        [triangle_points, triangle_points[0, :][np.newaxis, :]], axis=0
    )
    axes.plot(*points.T, **kwargs)
    return axes


def plot_line(ln0, ln, length=[-5, 5], axes=None, **kwargs):
    if "color" not in kwargs:
        kwargs["color"] = "k"

    if axes is None:
        axes = plt.figure().add_subplot(projection="3d")

    axes.plot(*np.vstack([ln0, ln0 + ln]).T, **kwargs)
    axes.plot(*np.vstack([ln0, ln0 + ln * length[0]]).T, linestyle=":", **kwargs)
    axes.plot(*np.vstack([ln0, ln0 + ln * length[1]]).T, linestyle=":", **kwargs)
    axes.plot(*(ln0 + ln), ".", **kwargs)

    return axes


def plot_point(point: np.ndarray, axes=None, **kwargs):
    # for a point w shape (3,)
    if "color" not in kwargs:
        kwargs["color"] = "k"

    if axes is None:
        axes = plt.figure().add_subplot(projection="3d")

    point = point[np.newaxis, :]
    axes.plot(*point.T, ".", **kwargs)
    return axes


def plot_mesh(vertices, connectivity_list, n=200, axes=None, **kwargs):
    """Plots brain surface triangulation."""
    if axes is None:
        axes = plt.figure().add_subplot(projection="3d")

    if "color" not in kwargs:
        kwargs["color"] = "k"

    if n is None:
        ix = np.arange(connectivity_list.shape[0])
    else:
        # ix = np.random.randint(connectivity_list.shape[0], size=n)
        ix = np.arange(connectivity_list.shape[0])
        np.random.shuffle(ix)
        ix = ix[:n]

    for i in ix:
        face = vertices[connectivity_list[i]]
        plot_triangle(face, axes=axes, lw=0.2, color="k")

    return axes


def plot_brain_surface_points(
    brain_surface_points, ds=4, axes=None, labels=None, atlas=None
):
    if axes is None:
        axes = plt.figure().add_subplot(projection="3d")

    if labels is not None:
        colors = np.zeros((int(brain_surface_points.shape[0] / ds), 4))
        color_mapping = atlas.regions.rgba
        for i, point in enumerate(brain_surface_points[::ds]):
            label = atlas.get_labels(point / 1e6, mode="clip")
            colors[i] = color_mapping[atlas.regions.id2index(label)[1][0][0]] / 255
    else:
        colors = None

    axes.scatter(*brain_surface_points[::ds, :].T, ".", s=1, alpha=0.2, c=colors)
    axes.set_aspect("equal")
    return axes


def plot_mlap_im_scatter(mlap_im, ref_surface_points, cs2d):
    # mlap_im: a (n_px, 3) array, columns are ml, ap, im (im = image value at coordinates)
    fig, axes = plt.subplots()
    ml, ap, im = mlap_im.T
    kwargs = dict(zip(["vmin", "vmax"], np.percentile(im, (5, 95))))
    axes.scatter(ap, ml, c=im, **kwargs)  # flipping the axes here
    axes.set_xlabel("ap")
    axes.set_ylabel("ml")
    kwargs = dict(color="w", lw=1)
    axes.axhline(0, **kwargs)
    axes.axvline(0, **kwargs)
    axes.set_xlabel("AP (µm)")
    axes.set_ylabel("ML (µm)")
    axes.invert_xaxis()
    axes.set_aspect("equal")

    # plot the points in this image
    points = np.array([point["coords"] for point in ref_surface_points["points"]])
    points = points[:, [1, 0]]
    points_ = cs2d.transform(points, "image", "mlap")
    axes.scatter(points_[:, 1], points_[:, 0], s=5, color="r")
    return axes


def plot_mlap_im_matshow(mlap_im, orig_shape, cs2d, ref_surface_points=None):
    im, extent = reshape_to_image(mlap_im, orig_shape)

    # extent order is (left, right, bottom, top)
    vmin, vmax = np.percentile(im, (5, 95))
    fig, axes = plt.subplots()
    axes.matshow(im, extent=extent, vmin=vmin, vmax=vmax, cmap="gray")

    # plot the points in this image
    if ref_surface_points:
        points = np.array([point["coords"] for point in ref_surface_points["points"]])
        points = points[:, [1, 0]]
        points_ = cs2d.transform(points, "image", "mlap")
        axes.scatter(points_[:, 1], points_[:, 0], s=25, color="m")

    kwargs = dict(color="w", lw=1, linestyle=":")
    axes.axhline(0, **kwargs)
    axes.axvline(0, **kwargs)
    axes.set_xlabel("AP (µm)")
    axes.set_ylabel("ML (µm)")
    axes.xaxis.set_label_position("top")
    return axes


def labels_to_image(mlap, labels, ds, orig_shape):
    # orig shape is here
    # n_px = np.array(ref_img.shape[1:])  # this is in (y, x) == (ml, ap)

    ml, ap = mlap[::ds].T
    # these are the vectors to span the interpolation grid.
    # the same number of points as the original image
    ml_vec = np.linspace(np.min(ml), np.max(ml), orig_shape[0])[::-1]
    ap_vec = np.linspace(np.min(ap), np.max(ap), orig_shape[1])[::-1]
    ap_grid, ml_grid = np.meshgrid(ap_vec, ml_vec)

    labels_image = griddata(
        np.stack([ml, ap], axis=1), labels, (ml_grid, ap_grid), method="nearest"
    )
    labels_image[np.isnan(labels_image)] = 0

    # extent
    extent = (np.max(ap), np.min(ap), np.min(ml), np.max(ml))

    return labels_image, extent
