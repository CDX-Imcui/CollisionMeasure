conda create -n gaussian_splatting python=3.7
conda activate gaussian_splatting

pip install torch==1.12.1+cu116 torchaudio==0.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm opencv-python  joblib plyfile matplotlib  scikit-learn open3d fastapi numpy  python-docx==0.8.11 pillow 

python MyApplication.py
![演示图](_internal/1.png)

![演示图](_internal/2.png)
