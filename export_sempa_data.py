#!/usr/bin/env python

import getopt
import glob
import io
import sys

import numpy as np

import pynistview_utils as pn


def process_file(file_name):
    print(f'Processing {file_name}...')

    data, axis = pn.image_data(file_name)
    scale = pn.get_scale(file_name)

    base_file_name = file_name[0:file_name.find('.sempa')]

    np.save(base_file_name, data)
    np.savetxt(base_file_name + '.csv', data, delimiter=',')

    try:
        f = open(base_file_name + '.txt', 'w')
        f.write(f'scale: {scale}')
    except Exception as ex:
        print(type(ex))
        print(ex)
    finally:
        f.close()


def main(argv):
    num_args = len(argv)

    if num_args == 1:
        print(f'usage: {argv[0]} <sempa_files>')
    else:
        for i in range(1, len(argv)):
            if argv[i].endswith('.sempa'):
                process_file(argv[i])

    print('Done.')


if __name__ == '__main__':
    main(sys.argv)
