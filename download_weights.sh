#!/usr/bin/env bash


mkdir -p relso_model_weights
cd relso_model_weights

echo "downloading model weights config"
# download model weight config
gdown https://drive.google.com/drive/u/0/folders/10FrNlvfU-mZYJVg2qjJxYgUgzSoSh6ln --folder

mv relso_trained_models/* .
rm -r relso_trained_models

unzip model_embeddings.zip
unzip trained_models.zip

cd ..