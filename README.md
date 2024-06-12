# Diff-Oracle



## Data Preparation

The oracle-241 dataset is publicly available at [STSN](https://github.com/wm-bupt/STSN), and the [OBC306](https://doi.org/10.1109/ICDAR.2019.00114) dataset can be downloaded at [OBC306](https://jgw.aynu.edu.cn/home/down/detail/index.html?sysid=16). However, the [Oracle-AYNU](https://doi.org/10.1109/ICDAR.2019.00057) dataset is provided by other institutions and is not publicly accessible due to institutional restrictions.

Pseudo handprinted oracle character images can be generated via [CUT](https://github.com/taesungp/contrastive-unpaired-translation) by translating scanned data into handprinted data.

Two JSON files: one is for training, and the other is for generation. 
 - For training, JSON file includes prompts, target (i.e., content/handprinted) and source (i.e., style/scanned) filenames, is presented below:
![p](./github_page/json_example.JPG)
 - For generation, JSON file includes prompts, labels, target (i.e., content/handprinted) and source (i.e., style/scanned) filenames, is presented below:
![p](./github_page/json_example_generation.JPG)




## Pretrain Model

The pretrain model "control_sd15_ini.ckpt" can be generated at [ControlNet-Step 3](https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md)




## Training

Once data and the pretrained model are prepared, run the training code.

 - For Oracle-241, run the file `train_style_oracle241.py` (the first stage training); run the file `train_content_oracle241.py` (the second stage training);
 - For OBC306, run the file `train_style_obc.py` (the first stage training); run the file `train_content_obc.py` (the second stage training);




## Generation

Once the model is well trained, run the generation code.

 - For Oracle-241, run the file `test_oracle241.py`
 - For OBC306, run the file `test_obc.py`
