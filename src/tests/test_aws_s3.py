import unittest
import os
import shutil
import boto3
from handlers.aws_s3 import download_files_from_s3_dir, upload_file_to_s3_dir


S3_BUCKET_NAME = 'idealo-hotel-image-assessment'
S3_FILE_PATH = 'ava_test'
SAVE_PATH = "./s3_test_download/"


class TestAmazonWebServicesS3(unittest.TestCase):

    def test_download_files_from_s3_dir(self):
        download_files_from_s3_dir(S3_BUCKET_NAME, S3_FILE_PATH, SAVE_PATH)

        directory_exist = os.path.exists(SAVE_PATH)
        all_files = [f for f in os.listdir(SAVE_PATH)]

        self.assertTrue(directory_exist)
        self.assertEqual(len(all_files), 12)
        shutil.rmtree(SAVE_PATH)  # remove downloaded directory

    # def test_upload_upload_file_to_s3_dir(self):
    #     S3_BUCKET_NAME = 'idealo-hotel-image-assessment'
    #     S3_FILE_PATH = 'test'
    #     SAVE_FILE = './test_images/42039.jpg'
    #
    #     upload_file_to_s3_dir(S3_BUCKET_NAME, S3_FILE_PATH, SAVE_FILE)
    #
    #     client = boto3.client('s3')
    #     expected = client.list_objects(Bucket="S3_BUCKET_NAME", Prefix='test/42039.jpg')
    #
    #     self.assertTrue(expected)
    #     client.delete_object(S3_BUCKET_NAME, Key='test/42039.jpg')

