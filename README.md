# OBBStacking

This repo is for the python implementation of [OBBStacking: An Ensemble Method for Remote Sensing Object Detection](https://arxiv.org/abs/2209.13369).

## Latest Updates

**2022/09/30**

First version released, only FAIR1M data format (results and labels stored in xmls) supported for now.

## Installation

```bash
pip install -r requirements.txt
```
## Usage

### FAIR1M

For FAIR1M data format, assume such folder structure for results from both the validation set and the test set:

```markdown
result_root_dir/
├── model_1_val/
│   └── test/
│       ├── 1.xml
│       ├── 2.xml
│       └── ...
├── model_2_val/
│   └── test/
│       ├── 1.xml
│       ├── 2.xml
│       └── ...
├── model_1_test/
│   └── test/
│       ├── 1.xml
│       ├── 2.xml
│       └── ...
├── model_2_test/
│   └── test/
│       ├── 1.xml
│       ├── 2.xml
│       └── ...
└── ...
```

To train the meta-learner, use the command below. Make sure to use the results from the validation set.
Note that parameter `output_name` should point to the folder of the groundtruth xmls.
The learned meta-learner will be saved to `weights_url`
```bash
python obbstacking.py "/path/to/image/folder/*.tif" --root_dir result_root_dir --input_names model_1_val model_2_val --output_name /path/to/ground/truth/folder/ --mode train --format FAIR --weights_url "weight.pkl"
```

To ensemble the test results, use the command below. Make sure to use the results from the test set.
The ensemble results will be saved in the folder `output_name` under the `root_dir` folder.
```bash
python obbstacking.py "/path/to/image/folder/*.tif" --root_dir result_root_dir --input_names model_1_test model_2_test --output_name "ensemble_result" --mode test --format FAIR --weights_url "weight.pkl"
```

### DOTA

Will be released soon.

## Citation

Please cite my article if you find the code or method helpful.
```text
@article{https://doi.org/10.48550/arxiv.2209.13369,
  doi = {10.48550/ARXIV.2209.13369},
  url = {https://arxiv.org/abs/2209.13369},
  author = {Lin, Haoning and Sun, Changhao and Liu, Yunpeng},
  title = {OBBStacking: An Ensemble Method for Remote Sensing Object Detection},
  publisher = {arXiv},
  year = {2022},
}
```
