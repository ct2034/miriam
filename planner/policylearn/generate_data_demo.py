import argparse
import logging
import pickle

from matplotlib import pyplot as plt

PLOT_FOVS_STR = "plot_fovs"


def plot_fovs(one_fov_data):
    frames = one_fov_data.shape[2]
    rows = int(frames / 2)
    assert rows * 2 == frames
    subplot_base = rows * 100 + 21  # two columns x rows

    for frame in range(frames):
        plt.subplot(subplot_base + frame)
        plt.imshow(one_fov_data[:, :, frame], cmap='gray')

    plt.show()


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='mode', choices=[
        PLOT_FOVS_STR, ])
    parser.add_argument(
        'fname_read_pkl', type=argparse.FileType('rb'))
    args = parser.parse_args()

    with open(args.fname_read_pkl.name, 'rb') as f:
        d = pickle.load(f)

    if args.mode == PLOT_FOVS_STR:
        for i in range(4):
            plot_fovs(d[i][0])
