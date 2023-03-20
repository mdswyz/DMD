# Decoupled Multimodal Distilling for Emotion Recognition, CVPR 2023

![](https://img.shields.io/badge/Platform-PyTorch-blue)
![](https://img.shields.io/badge/Language-Python-{green}.svg)
![](https://img.shields.io/npm/l/express.svg)

We propose a decoupled multimodal distillation (DMD) approach that facilitates flexible and adaptive crossmodal knowledge distillation. The key ingredients includes:
- The representation of each modality is decoupled into two parts, i.e., modality-irrelevant/-exclusive spaces. 
- We utilizes a graph distillation unit (GD-Unit) for each decoupled part so that each knowledge distillation can be performed in a specialized and effective manner.
- A GD-Unit consists of a dynamic graph where each vertice represents a modality and each edge indicates a dynamic knowledge distillation. 

In general,  the proposed GD paradigm provides a flexible knowledge transfer manner where the distillation weights can be automatically learned, thus enabling diverse crossmodal knowledge transfer patterns.


## The motivation.
<div align=center><img src="figure_1.png" width="50%"></img></div>

Motivation and main idea: (a) illustrates the significant emotion recognition discrepancies using unimodality, adapted from [Mult](https://github.com/yaohungt/Multimodal-Transformer). (b) shows the conventional cross-modal distillation. (c) shows our proposed DMD.

## The Framework.
![](figure2.png)
The framework of DMD.

## The learned graph edges.
![](edge.png)
Illustration of the graph edges in HomoGD and HeteroGD. In (a), $L \to A$ and $L \to V$ are dominated because the homogeneous language features contribute most and the other modalities perform poorly. In (b), $L \to A$, $L \to V$, and $V \to A$ are dominated.  $V \to A$ emerges because the visual modality enhanced its feature discriminability via the multimodal transformer mechanism in HeteroGD.

## Usage

### Prerequisites
- Python 3.8
- PyTorch 1.9.0
- CUDA 11.4

### Datasets
Data files (containing processed MOSI, MOSEI datasets) can be downloaded from [here](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing).
Please note that the meta information and the raw data are not available due to privacy of Youtube content creators. For more details, please follow the [official website](https://github.com/A2Zadeh/CMU-MultimodalSDK) of these datasets.

### Run the Codes

### Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{DMD,
  title={Decoupled Multimodal Distilling for Emotion Recognition},
  author={Yong Li and Yuanzhi Wang and Zhen Cui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```




