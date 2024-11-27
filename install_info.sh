wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
chmod +x Miniconda3-latest-Linux-x86_64.sh && \
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
export PATH="$HOME/miniconda/bin:$PATH" && \
conda create -n dicla python=3.10 -y && \
source $HOME/miniconda/bin/activate dicla && \
mkdir -p /home/projects/vokquant/data/dicla && \
git clone https://github.com/lorgu/accent-recog-slt2022 /home/projects/vokquant/accent-recog-slt2022 && \
cd /home/projects/vokquant/accent-recog-slt2022/ && \
pip install -q -r requirements.txt && \
pip install -q torchtext==0.13.1 torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
apt-get update && apt-get install -y pigz && \
tar --use-compress-program=pigz -xf /workspace/at_augmented.tar.gz -C /home/projects/vokquant/data/dicla/
