### Decoupled Multimodal Distilling for Emotion Recognition, CVPR 2022 

Human multimodal emotion recognition (MER) aims to perceive human emotions via language, visual and acoustic modalities. Despite the impressive performance of previous MER approaches, the inherent multimodal heterogeneities still haunt and the contribution of different modalities varies significantly. We propose a decoupled multimodal distillation (DMD) approach that facilitates flexible and adaptive crossmodal knowledge distillation. Specially, the representation of each modality is decoupled into two parts, i.e., modality-irrelevant/-exclusive spaces. DMD utilizes a graph distillation unit (GD-Unit) for each decoupled part so that each GD can be performed in a more specialized and effective manner. A GD-Unit consists of a dynamic graph where each vertice represents a modality and each edge indicates a dynamic knowledge distillation. Such GD paradigm provides a flexible knowledge transfer manner where the distillation weights can be automatically learned, thus enabling diverse crossmodal knowledge transfer patterns.

<img src="figure_1.png" width="50%"></img>

Motivation and main idea: (a) illustrates the significant emotion recognition discrepancies using unimodality, adapted from [Mult]([https://github.com/NVlabs/ffhq-dataset](https://github.com/yaohungt/Multimodal-Transformer)). (b) shows the conventional cross-modal distillation. (c) shows our proposed decoupled multimodal distillation (DMD) method.
