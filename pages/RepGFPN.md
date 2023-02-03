Feature pyramid network aims to aggregate different resolution features extracted from the backbone, which
    has been proven to be a critical and effective part of object detection [10, 30, 35]. The conventional FPN [10]
    introduces a top-down pathway to fuse multi-scale features.Considering the limitation of one-way information flow,
    PAFPN [35] adds an additional bottom-up path aggregation network, but with higher computational costs. BiFPN [30]
    removes nodes that only have one input edge, and adds skip link from the original input on the same level. In [16],
    Generalized-FPN (GFPN) is proposed to serve as neck and achieves SOTA performance, as it can sufficiently
    exchange high-level semantic information and low-level spatial information. In GFPN, multi-scale features are
    fused in both level features in previous and current layers.

What’s more, the log2(n) skip-layer connections provide more effective information transmission that can scale into
    deeper networks. When we directly replace modifiedPANet with GFPN on modern YOLO-series models, we achieved higher precision. However, the latency of GFPNbased model is much higher than modified-PANet-based
    model. 
    
By analyzing the structure of GFPN, the reason can be attributed to the following aspects: 
    
1) feature maps with different scales share the same dimension of channels; 
2) The operation of queen-fusion can not meet the requirement for real-time detection model; 
3) the convolution-based cross-scale feature fusion is not efficient.