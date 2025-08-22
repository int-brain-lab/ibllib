import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as mcolors


def plot_path(points, closed=True, axes=None, **kwargs): ...


def plot_plane(p0, n, size, axes=None, **kwargs):
    # calculate the edge points
    # plot a closed path
    ...


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


def plot_line(l0, l, length=[-5, 5], axes=None, **kwargs):
    if "color" not in kwargs:
        kwargs["color"] = "k"

    if axes is None:
        axes = plt.figure().add_subplot(projection="3d")

    axes.plot(*np.vstack([l0, l0 + l]).T, **kwargs)
    axes.plot(*np.vstack([l0, l0 + l * length[0]]).T, linestyle=":", **kwargs)
    axes.plot(*np.vstack([l0, l0 + l * length[1]]).T, linestyle=":", **kwargs)
    axes.plot(*(l0 + l), ".", **kwargs)

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


# def plot_labels_and_ref_image(
#     mlap_im,
#     ref_img,
#     labels,
#     ds,
#     orig_shape,
#     atlas,
#     points_imaging_im,
#     center_mlap,
#     cs2d,
# ):
#     im_rs = reshape_to_image(mlap_im, ref_img.shape[1:])[0]
#     labels_image, extent = labels_to_image(mlap_im[:, :-1], labels, ds, orig_shape)

#     fig, axes = plt.subplots(figsize=[10, 10])

#     for label in np.unique(labels):
#         ml_ix, ap_ix = np.where(labels_image == label)
#         im = np.zeros(labels_image.shape) * np.nan
#         im[ml_ix, ap_ix] = im_rs[ml_ix, ap_ix]
#         main_color = atlas.regions.rgba[label] / 255
#         # saturate
#         main_color = mcolors.rgb_to_hsv(main_color[:-1])
#         main_color[2] *= 1.2
#         main_color = np.clip(main_color, 0, 1)
#         main_color = mcolors.hsv_to_rgb(main_color)
#         cmap = mcolors.LinearSegmentedColormap.from_list(
#             "custom_colormap", ["black", main_color]
#         )
#         kwargs = dict(
#             vmin=np.percentile(im_rs, 5),
#             vmax=np.percentile(im_rs, 90),
#             cmap=cmap,
#             extent=extent,
#         )
#         axes.matshow(im, **kwargs)

#     # smooth boundaries
#     from skimage import feature

#     all_edges = np.zeros_like(labels_image)
#     for label in np.unique(labels_image):
#         edges = feature.canny(labels_image == label, sigma=3).astype("float64")
#         all_edges += edges
#     all_edges[all_edges > 0] = 1
#     from scipy import ndimage

#     # all_edges = ndimage.binary_dilation(all_edges.astype('bool'), iterations=1).astype('float64')
#     all_edges[all_edges == 0] = np.nan

#     # cut out the circle
#     from scipy import spatial

#     good_ix = (
#         spatial.distance_matrix(points_imaging_im[:, :-1], center_mlap[np.newaxis, :])
#         < 2500
#     )
#     mask = good_ix.reshape(all_edges.shape, order="C")
#     all_edges[~mask] = np.nan
#     axes.matshow(all_edges, cmap="gray_r", extent=extent, alpha=1.0)
#     # axes.matshow(all_edges, cmap="gray_r", vmax=1, extent=extent, alpha=0.5)

#     # hard boundaries
#     # from skimage import filters
#     # edges = filters.sobel(labels_image).astype("float64")
#     # edges[edges == 0] = np.nan
#     # edges[~np.isnan(edges)] = 1
#     # axes.matshow(edges, cmap="gray", extent=extent, alpha=1.0)

#     # plot the approximate imaging window
#     from matplotlib.patches import Circle

#     circle = Circle(
#         (center_mlap[1], center_mlap[0]),
#         2500,
#         facecolor="none",
#         edgecolor="w",
#         alpha=1,
#         lw=2,
#     )
#     axes.add_patch(circle)

#     # get center coordinates for each present brain region
#     for label in np.unique(labels):
#         ml_ix, ap_ix = np.where(labels_image == label)
#         points = cs2d.transform(np.stack([ml_ix, ap_ix], axis=1), "pixel", "mlap")
#         x, y = np.median(points, axis=0)
#         if spatial.distance.euclidean(center_mlap, [x, y]) < 2500:
#             axes.text(
#                 y,
#                 x,
#                 atlas.regions.index2acronym(label),
#                 ha="center",
#                 va="center",
#                 color="w",
#             )

#     axes.set_xlabel("AP (µm)")
#     axes.set_ylabel("ML (µm)")
#     axes.axhline(0, c="w", lw=1, linestyle=":")
#     axes.axvline(0, c="w", lw=1, linestyle=":")
#     axes.xaxis.set_label_position("top")

#     plt.grid(which="major", color="w", lw=1, alpha=0.6)
#     plt.grid(which="minor", color="w", lw=0.85, alpha=0.4, linestyle=":")
#     plt.minorticks_on()
#     from matplotlib.ticker import MultipleLocator

#     axes.xaxis.set_minor_locator(MultipleLocator(250))
#     axes.yaxis.set_minor_locator(MultipleLocator(250))

#  adding FOVs
# if 0:
#     for i, fov in enumerate(raw_img_meta["FOV"]):
#         fov_corners = fov["MM"]
#         keys = ["topLeft", "topRight", "bottomRight", "bottomLeft", "topLeft"]
#         fov_mlap = np.array([fov_corners[key] for key in keys]) * 1e3
#         if i == 0:
#             lw = 2
#         else:
#             lw = 1

#         axes.plot(
#             fov_mlap[:, 1], fov_mlap[:, 0], lw=lw, color="r"
#         )  # the ususal ap/ml reversal


###


# def plot_labels_image():
#     labels_image, extent = labels_to_image(mlap, labels, ds, ref_img.shape[1:])
#     n_labels = atlas.regions.rgba.shape[0]
#     fig, axes = plt.subplots(figsize=[10, 10])

#     axes.matshow(
#         labels_image.astype("int64"),
#         cmap=cmap,
#         vmin=0,
#         vmax=n_labels,
#         extent=extent,
#         alpha=1.0,
#     )

#     axes.axhline(0, color="k", lw=1)
#     axes.axvline(0, color="k", lw=1)

#     plt.grid(which="major", color="k", lw=1, alpha=0.8)
#     plt.grid(which="minor", color="k", lw=0.5, linestyle=":")
#     plt.minorticks_on()
#     from matplotlib.ticker import MultipleLocator

#     axes.xaxis.set_minor_locator(MultipleLocator(250))
#     axes.yaxis.set_minor_locator(MultipleLocator(250))

#     # smooth boundaries
#     from skimage import feature

#     all_edges = np.zeros_like(labels_image)
#     for label in np.unique(labels_image):
#         edges = feature.canny(labels_image == label, sigma=3).astype("float64")
#         all_edges += edges
#     all_edges[all_edges > 0] = 1
#     from scipy import ndimage

#     all_edges = ndimage.binary_dilation(all_edges.astype("bool"), iterations=1).astype(
#         "float64"
#     )
#     all_edges[all_edges == 0] = np.nan
#     axes.matshow(all_edges, cmap="gray", extent=extent, alpha=1.0)

#     # hard boundaries
#     # from skimage import filters
#     # edges = filters.sobel(labels_image).astype("float64")
#     # edges[edges == 0] = np.nan
#     # edges[~np.isnan(edges)] = 1
#     # axes.matshow(edges, cmap="gray", extent=extent, alpha=1.0)

#     # plot the approximate imaging window
#     from matplotlib.patches import Circle

#     circle = Circle(
#         (center_mlap[1], center_mlap[0]),
#         2500,
#         facecolor="none",
#         edgecolor="k",
#         alpha=1,
#         lw=2,
#     )
#     axes.add_patch(circle)

#     # get center coordinates for each present brain region
#     for label in np.unique(labels):
#         ml_ix, ap_ix = np.where(labels_image == label)
#         points = cs2d.transform(np.stack([ml_ix, ap_ix], axis=1), "pixel", "mlap")
#         x, y = np.median(points, axis=0)
#         axes.text(y, x, atlas.regions.index2acronym(label), ha="center", va="center")

#     axes.set_xlabel("AP (µm)")
#     axes.set_ylabel("ML (µm)")
#     axes.xaxis.set_label_position("top")
