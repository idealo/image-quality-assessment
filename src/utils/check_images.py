
import os
import glob
import argparse
from multiprocessing import Pool
from utils.utils import load_image


'''
checks each image in image dir whether it can be loaded
prints its name if not, need to remove manually from labels files
'''

NUM_WORKERS = 8


def pool_async(function, args_list, num_processes):
    pool = Pool(processes=num_processes)
    results = [pool.apply_async(function, args=args) for args in args_list]
    results = [res.get() for res in results]
    pool.close()
    pool.join()
    return results


def validate_image(image_file):
    valid = False
    try:
        img = load_image(image_file, (224, 224))
        assert img.shape == (224, 224, 3), '{}, {} image shape not correct'.format(image_file, img.shape)
        valid = True
    except OSError as e:
        print('{} can not be loaded'.format(image_file))
    return valid


def main(image_dir):
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

    print('checking {} images in {} ...'.format(len(image_files), image_dir))
    image_files = [[image_file] for image_file in image_files]
    results = pool_async(validate_image, image_files, NUM_WORKERS)
    print('{} images corrupt'.format(results.count(False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image-dir', required=True)

    args = parser.parse_args()
    main(**args.__dict__)
