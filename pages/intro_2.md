Especially, YOLOv5/6/7 [17, 31, 33],YOLOX [9] and PP-YOLOE [37] have achieved significant AP-Latency trade-offs on COCO,
    making YOLO series object detection methods widely used in the industry.
    Although object detection has achieved great progress,there are still new techs that can be brought in to further
    improve performance.

1.  Firstly, the network structure plays a critical role in object detection. Darknet holds a dominant position in the early stages of YOLO history [1, 9, 23–25, 31].

2. Recently, some works have investigated other efficient networks for their detectors, i.e., YOLOv6 [17] and YOLOv7 [33]. 
However, these networks are still manually designed. Thanks to the development of the Neural Architecture Search (NAS), there are many detection-friendly
network structures found through the NAS techs [4,15,29].

Therefore, we take advantage of the NAS techs and import MAE-NAS [29]for our DAMO YOLO.