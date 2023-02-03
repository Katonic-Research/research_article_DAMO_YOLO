In this report, we present a fast and accurate object
detection method dubbed DAMO-YOLO, which achieves
higher performance than the state-of-the-art YOLO series. DAMO-YOLO is extended from YOLO with some new
technologies, including Neural Architecture Search (NAS),
efficient Reparameterized Generalized-FPN (RepGFPN), a
lightweight head with AlignedOTA label assignment, and
distillation enhancement. In particular, we use MAE-NAS,
a method guided by the principle of maximum entropy,
to search our detection backbone under the constraints
of low latency and high performance, producing ResNetlike / CSP-like structures with spatial pyramid pooling and
focus modules. In the design of necks and heads, we
follow the rule of “large neck, small head”. We import
Generalized-FPN with accelerated queen-fusion to build
the detector neck and upgrade its CSPNet with efficient
layer aggregation networks (ELAN) and reparameterization. Then we investigate how detector head size affects
detection performance and find that a heavy neck with
only one task projection layer would yield better results.
In addition, AlignedOTA is proposed to solve the misalignment problem in label assignment. And a distillation
schema is introduced to improve performance to a higher
level. Based on these new techs, we build a suite of
models at various scales to meet the needs of different scenarios, i.e., DAMO-YOLO-Tiny/Small/Medium. They can
achieve 43.0/46.8/50.0 mAPs on COCO with the latency of
2.78/3.83/5.62 ms on T4 GPUs respectively