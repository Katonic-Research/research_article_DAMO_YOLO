# research_article_DAMO_YOLO

This app is about the research articles for **DAMO-YOLO : A Report on Real-Time Object Detection Design**

## Description

This article presents the DAMO-YOLO object identification approach, which outperforms the cutting-edge YOLO series in terms of speed and accuracy. A few new innovations, such as Neural Architecture Search (NAS), effective Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement, are added to DAMO-YOLO in comparison to YOLO. Uses MAE-NAS, a technique based on the maximisation of entropy, in particular to search detection backbones under the restrictions of low latency and high performance, producing ResNet-like / CSP-like structures with spatial pyramid pooling and focus modules. Create a collection of models at various scales, such as DAMO-YOLO-Tiny/Small/Medium, based on the latest technologies to satisfy the demands of diverse scenarios. With a latency of 2.78/3.83/5.62 ms on T4 GPUs, they can reach 43.0/46.8/50.0 mAPs on COCO.
