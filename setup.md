
Create environment and install dependencies
```sh
conda env create -n vl -y python=3.9

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install packaging
pip install -r InternVideo2/multi_modality/requirements.txt


# Need for flash-attention
wget https://github.com/Dao-AILab/flash-attention/archive/refs/tags/v2.1.1.zip
unzip v2.1.1.zip
rm v2.1.1.zip
cd flash-attention-2.1.1

export CUDA_HOME=/usr/local/cuda-11.7
conda install gxx_linux-64==11.2.0

cd csrc/fused_dense_lib && pip install .
cd ../..
cd csrc/layer_norm && pip install .

# Need PEFT
pip install peft
```