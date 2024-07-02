# Weakly-Supervised Medical Image Segmentation with Gaze Annotations
This is the PyTorch implementation of our MICCAI 2024 paper ["Weakly-Supervised Medical Image Segmentation with Gaze Annotations"]() by Yuan Zhong, Chenhui Tang, Yumeng Yang, Ruoxi Qi, Kang Zhou, Yuqi Gong, [Pheng-Ann Heng](https://www.cse.cuhk.edu.hk/~pheng/), [Janet Hsiao](https://jhhsiao.people.ust.hk/)\*, and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/)\* (\* denotes corresponding authors).

## Abstract

> Eye gaze that reveals human observational patterns has increasingly been incorporated into solutions for vision tasks. Despite recent explorations on leveraging gaze to aid deep networks, few studies exploit gaze as an efficient annotation approach for medical image segmentation which typically entails heavy annotating costs. In this paper, we propose to collect dense weak supervision for medical image segmentation with a gaze annotation scheme. To train with gaze, we propose a multi-level framework that trains multiple networks from discriminative human attention, simulated with a set of pseudo-masks derived by applying hierarchical thresholds on gaze heatmaps. Furthermore, to mitigate gaze noise, a cross-level consistency is exploited to regularize overfitting noisy labels, steering models toward clean patterns learned by peer networks. The proposed method is validated on polyp and prostate segmentation tasks using two public medical datasets. Our experiments demonstrate that gaze annotation outperforms previous label-efficient annotation schemes in terms of both performance and annotation time. 

## Features

- Publicly available gaze data for medical image segmentation as extension for the [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) and [NCI-ISBI](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/) datasets.
- A general plug-in framework for weakly-supervised medical image segmentation using gaze annotations.

## Gaze Dataset

Coming soon.

## Getting Started

#### Installation

1. Download from GitHub

   ```bash
   git clone https://github.com/med-air/GazeSup.git
   cd GazeSup
   ```

2. Create conda environment

   ```bash
   conda env create -f environment.yaml
   conda activate gaze
   ```

#### Preparing Gaze Annotation

Coming soon.

#### Running Experiments

```bash
python run.py -m [supervision_mode] --data [dataset] --model [backbone] -bs [batch_size] \
    --exp_path [experiment_path] --root [dataset_path] --spatial_size [image_size] --in_channels [image_channels] \
    --opt [optimizer] --lr [base_lr] --max_ite [maximum_iterations] --num_levels [num_levels] \
    --cons_mode [cons_mode] --cons_weight [cons_weight]
```

We provide the scripts of reproducing our experiments on the Kvasir-SEG and NCI-ISBI datasets with our gaze annotation [here](.\scripts). For more details on the arguments, please refer to [parse_args.py](.\parse_args.py). We also provide the model checkpoints for the experiments as listed below (Dice is the evaluation metric).

|           |                      Kvasir-SEG (Polyp)                      |                     NCI-ISBI (Prostate)                      |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Our paper |                            77.80                             |                            77.64                             |
| Released  | 77.67<br />[[script]](./scripts/gazesup_kvasir_2_levels.sh) [checkpoint] | 79.20<br />[[script]](./scripts/gazesup_prostate_2_levels) [checkpoint] |

## Contact

If you have any questions, please feel free to leave issues here, or contact [Yuan Zhong](mailto:yuanzhong@link.cuhk.edu.hk).

## Citation

Coming soon.