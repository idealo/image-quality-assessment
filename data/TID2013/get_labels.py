
import argparse
import numpy as np
import pandas as pd
from src.utils.utils import save_json
from maxentropy.skmaxent import MinDivergenceModel


# the maximised distribution must satisfy the mean for each sample
def get_features():
    def f0(x):
        return x

    return [f0]


def get_max_entropy_distribution(mean):
    SAMPLESPACE = np.arange(10)
    features = get_features()

    model = MinDivergenceModel(features, samplespace=SAMPLESPACE, algorithm='CG')

    # set the desired feature expectations and fit the model
    X = np.array([[mean]])
    model.fit(X)

    return model.probdist()


def get_dataframe(mean_raw_file):
    df = pd.read_csv(mean_raw_file, header=None, sep=' ')
    df.columns = ['mos', 'id']
    return df


def parse_raw_data(df):
    samples = []
    for i, row in df.iterrows():
        max_entropy_dist = get_max_entropy_distribution(row['mos'])
        samples.append({'image_id': row['id'].split('.')[0], 'label': max_entropy_dist.tolist()})

    return samples


def main(target_file, source_file_mean):
    df = get_dataframe(source_file_mean)
    samples = parse_raw_data(df)
    save_json(samples, target_file)
    print('Done! Saved JSON at {}'.format(target_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sfm', '--source-file-mean', help='file path of raw mos_with_names file', required=True)
    parser.add_argument('-tf', '--target-file', help='file path of json labels file to be saved', required=True)

    args = parser.parse_args()
    main(**args.__dict__)
