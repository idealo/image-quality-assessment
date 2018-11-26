FROM tensorflow/serving:latest

WORKDIR /tf_serving

# copy project files
COPY tfs_models/mobilenet_aesthetic /models/mobilenet_aesthetic/1
COPY tfs_models/mobilenet_technical /models/mobilenet_technical/1
COPY tf_serving_models.cfg /tf_serving/tf_serving_models.cfg

EXPOSE 8500
ENTRYPOINT []

CMD ["tensorflow_model_server" ,"--port=8500", "--model_config_file=tf_serving_models.cfg"]