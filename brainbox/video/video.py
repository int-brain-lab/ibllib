import numpy as np


def frame_diff(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    if frame1.shape != frame2.shape:
        raise ValueError('Frames must have the same shape')
    diff32 = np.float32(frame1) - np.float32(frame2)
    if frame1.ndim == 3:
        norm32 = (np.sqrt(diff32[:, :, 0] ** 2 + diff32[:, :, 1] ** 2 + diff32[:, :, 2] ** 2) /
                  np.sqrt(255 ** 2 * 3))
    else:
        norm32 = np.sqrt(diff32 ** 2 * 3) / np.sqrt(255 ** 2 * 3)
    return np.uint8(norm32 * 255)


def frame_diffs(frames):
    # Ensure array
    # frames = np.array(frames)
    # frames = np.array(
    #     [np.ones((5, 9, 3)), np.zeros((5, 9, 3)), np.ones((5, 9, 3)), np.zeros((5, 9, 3))],
    #     dtype=np.int32)
    df = np.empty((len(frames) - 2, *frames[0].shape[:2]), dtype=np.uint8)
    for i in range(len(frames) - 2):
        df[i] = frame_diff(frames[i], frames[i + 2])
        # dist = cv2.GaussianBlur(dist, (9, 9), 0)
        # calculate st dev test
        # _, stDev = cv2.meanStdDev(dist)
        # thresh.append(bool(stDev > sd_thresh))

    # Feature scaling
    df_ = df.sum(axis=(1, 2))
    df_ = (df_ - np.min(df_)) / (np.max(df_) - np.min(df_))
    return df_
