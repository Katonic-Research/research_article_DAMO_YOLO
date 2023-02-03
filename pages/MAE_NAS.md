MAE-NAS is a heuristic and training-free neural architecture search method without supernet dependence
    and can be utilized to archive backbones at different scales.It can produce ResNet-like / CSP-like structures with spatial
    pyramid pooling and focus modules. 

Secondly, it is crucial for a detector to learn sufficient fused information between high-level semantic and low-level spatial features, which
    makes the detector neck to be a vital part of the whole framework. The importance of neck has also been discussed
    in other works [10, 16, 30, 35]. Feature Pyramid Network (FPN) [10] has been proved effective to fuse multi-scale
    features. Generalized-FPN (GFPN) [16] improves FPN with a novel queen-fusion.

In DAMO-YOLO, we design a Reparameterized Generalized-FPN (RepGFPN). It is based
    on GFPN but involved in an accelerated queen-fusion, the efficient layer aggregation networks (ELAN) and reparameterization.

To strike the balance between latency and performance,we conducted a series of experiments to verify the importance of the 
    neck and head of the detector and found that *”large neck, small head” would lead to better performance*.
    Hence, we discard the detector head in previous YOLO series works [1, 9, 23–25, 31, 37], but only left a task
    projection layer. The saved calculations are moved to the neck part. Besides the task projection module, there is no
    other training layer in the head, so we named our detector head as ZeroHead. Coupled with our RepGFPN, ZeroHead
    achieves state-of-the-art performance, which we believe would bring some insights to other researchers.

In addition, the dynamic label assignment, such as OTA [8] and TOOD [7], is widely acclaimed and achieves
    significant improvement compared to the static label assignment [41]. However, the misalignment problem is still
    unsolved in these works. We propose a better solution called AlignOTA to balance the importance of classification
    and regression, which can partly solve the problem.At last, Knowledge Distillation (KD) has been proved
    effective in boosting small models by the larger model supervision. This tech does exactly fit the design of realtime object detection. Nevertheless, applying KD on YOLO
    series sometimes can not achieve significant improvements as hyperparameters are hard to optimize and features carry
    too much noise. 
    
In our DAMO-YOLO, we first make distillation great again on models of all sizes, especially on
    small ones.As shown in Fig.1, with the above improvements, we proposed a series of models that exceed the state of the
    arts by a large margin, e.g., **the DAMO-YOLO-S model achieves 46.8 mAP and outperforms YOLOv6-S 43.4 mAP
    and YOLOE-S 43.1 mAP, while its latency close to these models.**

In summary, the contributions are three-fold:

1. This paper proposes a new detector called DAMOYOLO, which extends from YOLO but with more new
techs, including MAE-NAS backbones, RepGFPN neck, ZeroHead, AlignedOTA and distillation enhancement.

2. DAMO-YOLO outperforms the state-of-the-art detectors (e.g. YOLO series) on public COCO datasets.

3. A suite of models with various scales is presented in DAMO-YOLO (tiny/small/medium) to support different deployments. 
The code and pre-trained models are released at https://github.com/tinyvision/damoyolo, with ONNX and TensorRT supported.