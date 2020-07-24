import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

def plot_map_binary(truth, prediction):

    if truth.shape != prediction.shape:

        print("Error: Expecting truth and prediction arrays to have same size but received shapes {} and {}".format(str(truth.shape), str(prediction.shape)))
        return
    else:

        h, w = truth.shape[0], truth.shape[1]
        tp = np.zeros(truth.shape)
        fp = np.zeros(truth.shape)
        fn = np.zeros(truth.shape)
        tn = np.zeros(truth.shape)

        for i in range(0, h):

            for j in range(0, w):

                if (truth[i, j] == 255 and prediction[i, j] == 255):

                    tp[i, j] = 1

                elif (truth[i, j] == 255 and prediction[i, j] == 0):

                    fn[i, j] = 1

                elif (truth[i, j] == 0 and prediction[i, j] == 255):

                    fp[i, j] = 1

                else:

                    tn[i, j] = 1

        iou = np.sum(tp) / (np.sum(tp) + np.sum(fp) + np.sum(fn))
        dice = (2 * np.sum(tp)) / (np.sum(tp) + np.sum(fp) + np.sum(tp) + np.sum(fn))

        plot_tp = ma.masked_values(tp * 100, 0)
        plot_fp = ma.masked_values(fp * 50, 0)
        plot_fn = ma.masked_values(fn, 0)

        plt.imshow(plot_tp, cmap = "brg", vmin = 0.1)
        plt.imshow(plot_fp, cmap = "brg", vmin = 0.1, vmax = 100)
        plt.imshow(plot_fn, cmap = "brg", vmin = 0.1, vmax = 90)

        plt.title("IoU: {}, Dice: {}".format(str(iou), str(dice)))

        plt.show()

def plot_map_multiclass(truth, prediction):

    print('not done ytet')

