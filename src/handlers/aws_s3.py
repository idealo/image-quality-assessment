import os
import boto3

s3 = boto3.resource('s3')


def download_file_from_s3(s3_bucket_name, s3_dir_path, s3_source_file, target_file):
    bucket = s3.Bucket(s3_bucket_name)
    bucket.Object(os.path.join(s3_dir_path, s3_source_file)).download_file(target_file)
    print('Downloaded {}'.format(target_file))


def download_files_from_s3_dir(s3_bucket_name, s3_dir_path, img_dir):
    ensure_dir_exists(img_dir)

    bucket = s3.Bucket(s3_bucket_name)

    s3_objects = iter(bucket.objects.filter(Prefix=s3_dir_path))
    next(s3_objects)

    print('Download from S3 Bucket {}/{}'.format(s3_bucket_name, s3_dir_path))

    for object in s3_objects:
        file_name = object.key.split("/")[-1]
        target_file = os.path.join(img_dir, file_name)
        bucket.download_file(object.key, target_file)
        print(target_file)

    print('Done.')


def upload_file_to_s3_dir(s3_bucket_name, s3_dir_path, path_to_file):
    s3.Object(s3_bucket_name, s3_dir_path+'/'+path_to_file.split("/")[-1]).put(Body=open(path_to_file, 'rb'))


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
