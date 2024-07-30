# Weakly-supervised Medical Image Segmentation with Gaze Annotations
This is the PyTorch implementation of our MICCAI 2024 paper ["Weakly-supervised Medical Image Segmentation with Gaze Annotations"](https://arxiv.org/abs/2407.07406) by Yuan Zhong, Chenhui Tang, Yumeng Yang, Ruoxi Qi, Kang Zhou, Yuqi Gong, [Pheng-Ann Heng](https://www.cse.cuhk.edu.hk/~pheng/), [Janet H. Hsiao\*](https://jhhsiao.people.ust.hk/), and [Qi Dou\*](https://www.cse.cuhk.edu.hk/~qdou/).

\* denotes corresponding authors.

## Abstract

> Eye gaze that reveals human observational patterns has increasingly been incorporated into solutions for vision tasks. Despite recent explorations on leveraging gaze to aid deep networks, few studies exploit gaze as an efficient annotation approach for medical image segmentation which typically entails heavy annotating costs. In this paper, we propose to collect dense weak supervision for medical image segmentation with a gaze annotation scheme. To train with gaze, we propose a multi-level framework that trains multiple networks from discriminative human attention, simulated with a set of pseudo-masks derived by applying hierarchical thresholds on gaze heatmaps. Furthermore, to mitigate gaze noise, a cross-level consistency is exploited to regularize overfitting noisy labels, steering models toward clean patterns learned by peer networks. The proposed method is validated on two public medical datasets of polyp and prostate segmentation tasks. We contribute a high-quality gaze dataset entitled **GazeMedSeg** as an extension to the popular medical segmentation datasets. To the best of our knowledge, this is the first gaze dataset for medical image segmentation. Our experiments demonstrate that gaze annotation outperforms previous label-efficient annotation schemes in terms of both performance and annotation time. 

![](./figures/schemes.png)

## Highlights

- Public gaze dataset **GazeMedSeg** for segmentation as extension for the [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) and [NCI-ISBI](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/) datasets.
- A general plug-in framework for weakly-supervised medical image segmentation using gaze annotations.

## Gaze Dataset

Please refer to [here](/GazeMedSeg) for detailed description of our GazeMedSeg dataset.

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

#### Preparing Datasets

>Note: You can download our preprocessed dataset [here](https://drive.google.com/drive/folders/1XjgQ27R8zT8ymOTXohgl8HXntPEUbIXj?usp=sharing), allowing you to skip this and the next step to reproduce our experiments.

- Download the [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) and [NCI-ISBI](https://www.cancerimagingarchive.net/analysis-result/isbi-mr-prostate-2013/) datasets from their official websites. For NCI-ISBI, please download the Training and Test sets.
- Preprocess the datasets. We only preprocess the NCI-ISBI dataset, extracting axial slices where the prostate is present. The relevant script is [here](/notebooks/preprocess/process_nci-isbi.ipynb).

#### Preparing Gaze Annotation

- Download the GazeMedSeg [here](https://drive.google.com/drive/folders/1-38bG_81OsGVCb_trI00GSqfB_shCUQG?usp=sharing), and put the files under the [`/GazeMedSeg`](/GazeMedSeg) folder.
- Generate gaze heatmaps and refined CRF maps using the scripts [here](notebooks/gaze_annotation). These scripts will create a `gaze` folder within the original dataset directory and generate gaze heatmaps and CRF maps there. The CRF maps will serve as pseudo-masks for gaze supervision.

#### Running Experiments

```bash
python run.py -m [supervision_mode] --data [dataset] --model [backbone] -bs [batch_size] \
    --exp_path [experiment_path] --root [dataset_path] --spatial_size [image_size] \
    --in_channels [image_channels] --opt [optimizer] --lr [base_lr] --max_ite [max_ite] \
    --num_levels [num_levels] --cons_mode [cons_mode] --cons_weight [cons_weight]
```

We provide the scripts of reproducing our experiments on the Kvasir-SEG and NCI-ISBI datasets with our gaze annotation [here](./scripts). For more details on the arguments, please refer to [parse_args.py](./parse_args.py). 

#### Checkpoints

We also provide the model checkpoints for the experiments as listed below (Dice is the evaluation metric).

|           |                      Kvasir-SEG (Polyp)                      |                     NCI-ISBI (Prostate)                      |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Our paper |                            77.80                             |                            77.64                             |
| Released  | 78.86<br />[[script]](./scripts/gazesup_kvasir_2_levels.sh) [[checkpoint]](https://drive.google.com/file/d/1e-P7TEOIDJ04edFy1Eix8bTl5ZRD3l-g/view?usp=sharing) | 79.20<br />[[script]](./scripts/gazesup_prostate_2_levels.sh) [[checkpoint]](https://drive.google.com/file/d/1wq60hlEPFhotwPM5tCxcFK-hjPBZ842L/view?usp=sharing) |

## Contact

If you have any questions, please feel free to leave issues here, or contact [Yuan Zhong](mailto:yuanzhong@link.cuhk.edu.hk).

## Citation

``` -->
@article{zhong2024weakly,
  title={Weakly-supervised Medical Image Segmentation with Gaze Annotations},
  author={Zhong, Yuan and Tang, Chenhui and Yang, Yumeng and Qi, Ruoxi and Zhou, Kang and Gong, Yuqi and Heng, Pheng Ann and Hsiao, Janet H and Dou, Qi},
  journal={International Conference on Medical Image Computing and Computer Assisted Intervention},
  year={2024}
}
