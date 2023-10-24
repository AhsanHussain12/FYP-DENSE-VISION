# Running CSRNet
1. Download Miniconda (You can skip this step if you already have Miniconda installed) from [here](https://docs.conda.io/en/latest/miniconda.html)
2. Clone the code from `https://github.com/abdumhmd/CSRNet` . You can do `git clone https://github.com/abdumhmd/CSRNet.git` or download the zip file from the repository.
3. Download the ShanghaiTech dataset from [here](https://iowastate-my.sharepoint.com/:u:/g/personal/abdu_iastate_edu/EQzcnQwcLzxPqa_X3cEnQcEB1GD77FRaDYjPp7KqBj5Ciw). Extract the zip file outside the repository folder.
4. On the command line, run `conda create --name csrnet python=3.8.13`
5. Run `conda activate csrnet`
6. Install the following libraries using the commands below:
   1. `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch`
   2. `conda install -c conda-forge opencv`
   3. `conda install -c conda-forge matplotlib`
   4. `conda install -c conda-forge tqdm`
   5. `pip install wandb`
   6. `pip install pytorch-lightning== 1.8.0`
   7. `pip install pillow`
   8. `pip install pandas`
   9. `pip install h5py`
