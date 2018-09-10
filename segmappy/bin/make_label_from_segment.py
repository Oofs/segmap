from __future__ import print_function
from builtins import input
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import matplotlib.colors as colors

import ensure_segmappy_is_installed
from segmappy import Dataset
from segmappy import Config
from segmappy.tools.classifiertools import get_default_dataset




def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()





configfile = "default_training.ini"
config = Config(configfile)
# tweak config parameters
config.folder = "loop_long"

WRITE_LABEL = True
AUTOWALLS = False
CLASS = 2
CLASSES = ["other", "car", "building"]
default_label = 0
# load dataset
dataset = get_default_dataset(config,config.folder)

# segments, _, ids, n_ids, features, matches, _ = dataset.load()
segments, _, ids, n_ids, features, matches= dataset.load_no_label()
labels = []
seg_ids = []

print("Type q and then ENTER to quit.")
for i in range(ids.size):
    # skip if it's not the last duplicate
    if i + 1 < ids.size and ids[i] == ids[i + 1]:
        continue


    fig = plt.figure(1)
    plt.clf()

    ax = fig.add_subplot(221, projection="3d")

    segment = segments[i]
    segment = segment - np.min(segment, axis=0)

    # Maintain aspect ratio on xy scale
    ax.set_xlim(0, np.max(segment[:, :]))
    ax.set_ylim(0, np.max(segment[:, :]))
    ax.set_zlim(0, np.max(segment[:, :]))

    x, y, z = np.hsplit(segment, segment.shape[1])
    
    ax.scatter(x,y,z, c=z.reshape(z.shape[0],))

    ax = fig.add_subplot(222)
    ax.scatter(x, y)
    ax.set_xlim(0, np.max(segment[:, :]))
    ax.set_ylim(0, np.max(segment[:, :]))

    ax = fig.add_subplot(223)
    ax.scatter(x, z)
    ax.set_xlim(0, np.max(segment[:, :]))
    ax.set_ylim(0, np.max(segment[:, :]))

    ax = fig.add_subplot(224)
    ax.scatter(y, z)
    ax.set_xlim(0, np.max(segment[:, :]))
    ax.set_ylim(0, np.max(segment[:, :]))



    plt.draw()
    plt.pause(0.001)

    while True:
        # autolabel
        max_x = max(segment[:, 0])
        min_x = min(segment[:, 0])
        max_y = max(segment[:, 1])
        min_y = min(segment[:, 1])
        max_z = max(segment[:, 2])
        min_z = min(segment[:, 2])

        if AUTOWALLS:

            dist = np.linalg.norm([max_x - min_x, max_y - min_y])

            if dist > 6:
                print(str(ids[i]) + " autolabeled as wall")
                label = 2
                labels.append(label)
                seg_ids.append(ids[i])
                break

        # consider user input
        print("dx:{} dy:{} dz:{}".format(max_x-min_x, max_y-min_y, max_z-min_z))
        label = input(str(ids[i]) + " label: ")
        
        if label in ["0", "1", "2"]:
            labels.append(label)
            seg_ids.append(ids[i])

        if not label:
            label = default_label
            labels.append(label)
            seg_ids.append(ids[i])
            break
        if label in ["0", "1", "2", "q"]:
            break

    if label == "q":
        break


# print(labels)
# print(range(len(labels)))
# print(ids)
if WRITE_LABEL:
    fp_labels = open(os.path.join(config.base_dir+config.folder, "labels_database.csv"), "w")
    num = 1
    for i in range(len(labels)):
        fp_labels.write(str(seg_ids[i]) + " " + str(labels[i]) + "\n")
