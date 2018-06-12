
import argparse
from handlers.config_loader import load_config
from handlers.aws_s3 import download_files_from_s3_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', help='file path of configuration file',
                        required=True)

    args = parser.parse_args()

    config = load_config(**args.__dict__)

    download_files_from_s3_dir(config['s3_bucket_name'], config['s3_file_path'], config['img_dir'])
