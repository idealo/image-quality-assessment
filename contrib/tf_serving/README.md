## Serving NIMA with TensorFlow Serving
TensorFlow versions of both the technical and aesthetic MobileNet models are provided,
along with the script to generate them from the original Keras files, under the `contrib/tf_serving` directory.

There is also an already configured TFS `Dockerfile` that you can use.

To get predictions from the aesthetic or technical model:
1. Build the NIMA TFS Docker image `docker build -t tfs_nima contrib/tf_serving`
2. Run a NIMA TFS container with `docker run -d --name tfs_nima -p 8500:8500 tfs_nima`
3. Install python dependencies to run TF serving sample client
```
virtualenv -p python3 contrib/tf_serving/venv_tfs_nima
source contrib/tf_serving/venv_tfs_nima/bin/activate
pip install -r contrib/tf_serving/requirements.txt
```
4. Get predictions from aesthetic or technical model by running the sample client
```
python -m contrib.tf_serving.tfs_sample_client --image-path src/tests/test_images/42039.jpg --model-name mobilenet_aesthetic
python -m contrib.tf_serving.tfs_sample_client --image-path src/tests/test_images/42039.jpg --model-name mobilenet_technical
```
