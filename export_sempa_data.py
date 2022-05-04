#!/usr/bin/env python

import os, sys, time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import pynistview_utils as pn

figsize = (3, 3)
dpi = 300


def process_file(file_name):
    print(f'Processing {file_name}...')

    data, axis = pn.image_data(file_name)
    scale = pn.get_scale(file_name)

    base_file_name = file_name[0:file_name.find('.sempa')]

    np.save(base_file_name, data)
    np.savetxt(base_file_name + '.csv', data, delimiter=',')
    
    process_image(base_file_name, data, scale)

    try:
        f = open(base_file_name + '.txt', 'w')
        f.write(f'm/pixel: {scale[0]}\n')
        f.write(f'Magnification: {scale[1]}x\n')
    except Exception as ex:
        print(type(ex))
        print(ex)
    finally:
        f.close()


def process_image(file_name, img, scale):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img, cmap='gray')
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.add_artist(ScaleBar(scale[0], box_alpha=0.8))

    fig.savefig(file_name)
    plt.close()


def main(argv):
    num_args = len(argv)
    
    counter = 0
    tick = time.perf_counter()
 
    if num_args == 1:
        print(f'usage: {argv[0]} <sempa_files>')
    else:
        for i in range(1, len(argv)):
            file = argv[i]
            if file.endswith('.sempa'):# and os.path.getsize(file) > 0:
                try:
                    process_file(argv[i])
                except Exception as ex:
                    print(type(ex))
                    print(file + ': ' + str(ex))

                counter += 1

    tock = time.perf_counter()
    duration = tock - tick

    print(
        f'Completed. Processed {counter} files in {duration:.1f} seconds ({duration / counter:.1f} seconds / file).')


if __name__ == '__main__':
    main(sys.argv)
