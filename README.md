<h2 align="center">NeRSP: Neural 3D Reconstruction for Reflective Objects with Sparse Polarized Images</h2>
<h4 align="center">
    <a href="https://yu-fei-han.github.io/homepage/"><strong>Yufei Han</strong></a>
    ·
    <a href="https://gh-home.github.io/"><strong>Heng Guo</strong></a>
    ·
    <strong>Koki Fukai</strong></a>
    ·
    <a href="https://sites.google.com/view/hiroaki-santo/"><strong>Hiroaki Santo</strong></a>
    ·
    <a href="https://camera.pku.edu.cn/"><strong>Boxin Shi</strong></a>
    ·
    <a href="http://cvl.ist.osaka-u.ac.jp/user/okura/"><strong>Fumio Okura</strong></a>
    ·
    <a href="https://zhanyuma.cn/"><strong>Zhanyu Ma</strong></a>
    ·
    <a href="https://sdmda.bupt.edu.cn/info/1061/1060.htm"><strong>Yunpeng Jia</strong></a>
</h3>
<h4 align="center"><a href="https://cvpr.thecvf.com/Conferences/2024">CVPR 2024 </a></h3>
<p align="center">
  <br>
    <a href="https://arxiv.org/abs/2406.07111">
      <img src='https://img.shields.io/badge/arXiv-Paper-981E32?style=for-the-badge&Color=B31B1B' alt='arXiv PDF'>
    </a>
    <a href='https://yu-fei-han.github.io/NeRSP-project/'>
      <img src='https://img.shields.io/badge/NeRSP-Project Page-5468FF?style=for-the-badge' alt='Project Page'></a>
</p>
<div align="center">
</div>

# Quick Start
Our code was tested on Ubuntu with Python 3.10, PyTorch 1.11 (2.x may meet trouble). Follow these steps to reproduce our environment and results.
```
conda create -n nersp python=3.10 -y
conda activate nersp
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
git clone https://github.com/PRIS-CV/NeRSP.git
```

# Dataset
We release the synthetic dataset **SMVP3D** and the real-wolrd dataset **RMVP3D**. All dataset is devided in original part and test part (6 views only). If you just want to train the results, pelase download the **test part** and put the case (object) folder under the new folder```./dataset```.

<details><summary> SMVP3D </summary>

SMVP3D has 5 objects rendering with different environment maps. All images are 512 * 512 size. You can download the [original part](https://drive.google.com/file/d/1gXJnJ9jfXD51_z0Ue1ME4nGxsXcxiw6h/view?usp=drive_link) and [test part](https://drive.google.com/drive/folders/1vsQ0pACWCVBwEnMI_gibwbTqbU00sDiE?usp=drive_link) from google drive.

</details>

<details><summary>RMVP3D</summary>

RMVP3D has 4 objects captured under room environments. The original images are 1024 * 1224 size. We train and test under 512 * 612 size. You can download the [original part](https://drive.google.com/file/d/1J1VN5J7t7tCea4HbybI0wJ6hOT8-bLwq/view?usp=sharing) and [test part](https://drive.google.com/drive/folders/1JrD1FtZF9Y5_fn9Cb2lED6aPGVCIFzIH?usp=drive_link) from google drive.

</details>

<details><summary>PANDORA</summary>

You can download the original dataset from [PANDORA](https://akshatdave.github.io/pandora/). The [test part](https://drive.google.com/drive/folders/1y5l0KZdtJB8o50xA3RjZJE-JI1jIsW20?usp=drive_link) for Vase and Owl tested in our method is under 512 * 612 size.
</details>

# Train
<details><summary> SMVP3D </summary>

After downloading the test part dataset of SMVP3D, you can run the code by:
```
python exp_runner.py --conf confs/wmask_ours_synthetic.conf --mode train --case snail 
```

</details>

<details><summary>RMVP3D</summary>

After downloading the test part dataset of RMVP3D, you can run the code by:
```
python exp_runner.py --conf confs/wmask_ours_real.conf --mode train --case shisa 
```
</details>

<details><summary>PANDORA</summary>

After downloading the test part dataset of PANDORA, you can run the code by:
```
python exp_runner.py --conf confs/wmask_pandora.conf --mode train --case owl 
```
</details>

# Validate
After training, run the code to output mesh and images.
```
 python exp_runner.py --conf <conf_file> --mode validate_mesh --case <case_name> 
```


# Acknowledgement
Our implementation is built from [NeuS](https://github.com/Totoro97/NeuS), [IDR](https://github.com/lioryariv/idr), [MVAS](https://github.com/xucao-42/mvas) and [PANDORA](https://github.com/akshatdave/pandora).


# Bibtex
```
@inproceedings{nersp2024yufei,
title = {NeRSP: Neural 3D Reconstruction for Reflective Objects with Sparse Polarized Images},
author = {Yufei, Han and Heng, Guo and Koki, Fukai and Hiroaki, Santo and Boxin, Shi and Fumio, Okura and Zhanyu, Ma and Yunpeng, Jia},
year = {2024},
booktitle = CVPR,
}
```
