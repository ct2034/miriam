import matplotlib.image as mpimg
import numpy as np
from matplotlib import pyplot as plt

from roadmaps.cvt.build.libcvt import CVT

if __name__ == "__main__":
    img_path = "roadmaps/odrm/odrm_eval/maps/plain.png"

    cvt = CVT()
    (nodes, t) = cvt.run(img_path, 100)
    img = mpimg.imread(img_path)
    nodes = np.array(nodes)

    # fig = plt.figure()
    # plt.imshow(img)
    # plt.plot(nodes[:, 1], nodes[:, 0], 'r.')

    # plt.axis('scaled')  # Ensure aspect ratio is maintained
    # plt.show()
