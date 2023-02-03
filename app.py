import streamlit as st
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(
    page_title='Interactive research article using Streamlit',  
    layout = 'centered', 
    initial_sidebar_state = 'auto'
)
st.markdown('*Research article*')
st.subheader('DAMO-YOLO : A Report on Real-Time Object Detection Design')
st.subheader("Authors")
st.warning('''
Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang, Xiuyu Sun†

 Alibaba Group
''')

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

st.header('Abstract')
abstract = read_markdown_file("pages/abstract.md")
st.info(abstract)

st.markdown('**Keywords:** *DAMO-YOLO*, *YOLO series*, *Neural Architecture Search (NAS)*, *Reparameterized Generalized-FPN (RepGFPN)*')

st.header('Introduction')
intro_markdown = read_markdown_file("pages/introduction.md")
st.write(intro_markdown, unsafe_allow_html=True)

st.image('image/rs-1.png')
st.caption('*Figure 1. Latency-accuracy trade-off of models for DAMO-YOLO and other state-of-the-art object detectors*')

intro_markdown_2 = read_markdown_file("pages/intro_2.md")
st.write(intro_markdown_2, unsafe_allow_html=True)

fig = go.Figure(data=[go.Table(
    header=dict(values=[' ', 'Backbone','AP','Latency(ms)'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['DAMO-YOLO-S', 'DAMO-YOLO-S', 'DAMO-YOLO-S', 'DAMO-YOLO-M','DAMO-YOLO-M'], # 1st column
                       [ 'CSP-Darknet', 'MAE-ResNet', 'MAE-CSP', 'MAE-ResNet','MAE-CSP'],
                       [ 44.9, 45.6,45.3,48.0,48.7],
                       [3.92, 3.83, 3.79, 5.64, 5.60]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)

st.caption('*Table 1. CSP-Darknet vs MAE-NAS Backbone under DAMOYOLO framework with different scales.*')

st.write('**MAE-NAS**')

MAE_NAS = read_markdown_file("pages/MAE_NAS.md")
st.write(MAE_NAS, unsafe_allow_html=True)

st.header('DAMO-YOLO')
st.write('''
In this section, we introduce each module of DAMOYOLO in detail, including Neural Architecture Search
(NAS) backbones, efficient Reparameterized GeneralizedFPN (RepGFPN) neck, ZeroHead, AlignedOTA label assignment and
distillation enhancement. The whole framework of DAMO-YOLO is displayed in Fig.2.
''')

st.image('image/rs-2.png')
st.caption('''Figure 2. Overview of the network architecture of DAMO-YOLO. 

1) MAE-NAS as backbone to extract multi-scale feature maps; 
2) Efficient RepGFPN as neck to refine and fuse high-level semantic and low-level spatial features;
3) ZeroHead is presented which only contains a task projection layer for each loss.''')

st.write('**MAE-NAS Backbone**')
with st.expander("Expand"):

    st.write('''
    Instead of scaling technology, we use MAE-NAS [29] to obtain optimal networks under different computational
    budgets. MAE-NAS constructs an alternative proxy based on information theory to rank initialized networks without
    training. Therefore, the search process only takes a few hours, which is much lower than the training costs.
    Following previous works [29], we design our backbones in the vanilla convolution network space with a new
    search block “k1kx”, which is similar to blocks used in Darknet-53 [25]. Meanwhile, inspired by YOLOv6 [17],
    we directly use the GPU inference latency, not FLOPs, as the target budget. After searching, we apply Spatial
    Pyramid Pooling (SPP) [12], Focus [31] and Cross Stage Partial (CSP) [34] modules into the final backbones.
    The performance comparisons of CSP-Darknet and our MAE-NAS backbones under our DAMO-YOLO with different scales are listed 
    in Table.1, which implies the effectiveness of MAE-NAS backbones. 
    
    In this table, “MAEResNet” means there are only SPP and Focus modules in the MAE-NAS backbones, and “MAE-CSP” means there
    are CSP modules in it as well. Besides, “S” (Small) and “M” (Medium) represent different scales of backbones.
    Considering the trade-off between performance and inference speed, we use “MAE-ResNet” in “T” (Tiny) and “S”
    scales and “MAE-CSP” in “M” scale in the final settings, as shown in Table.8.
    ''')

st.write('**Efficient RepGFPN**')
with st.expander("Expand"):

    MAE_NAS = read_markdown_file("pages/RepGFPN.md")
    st.write(MAE_NAS, unsafe_allow_html=True)
  
st.write(
    '''
    Based on GFPN, we propose a novel Efficient-RepGFPN to meet the design of real-time object detection, which
mainly consists of the following insights: 1) Due to the large difference in FLOPs from different scale feature maps,
it is difficult to control the same dimension of channels shared by each scale feature map under the constraint of
limited computation cost. Therefore, in the feature fusion of our neck, we adopt the setting of different scale feature
maps with different dimensions of channels. Performance with the same and different channels as well as precision
benefits from the Neck depth and width trade-offs are compared, Table.2 shows the results.
    '''
)

fig = go.Figure(data=[go.Table(
    header=dict(values=[' Depth', 'Width','Latency','FLOPs','AP'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[[2,2,3,3,4], # 1st column
                       [ (192, 192, 192), (128, 256, 512), (160, 160, 160), (96, 192, 384),(64, 128, 256)],
                       [ 3.53, 3.72, 3.91,3.83, 3.85],
                       [34.9, 36.1, 38.2, 37.8,  37.2],
                       [44.2, 45.1, 44.9, 45.6,  45.3]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)

st.caption('''Table 2. Ablation Study on the depth and width of our neck.
“Depth” denotes the repeat times on the bottleneck of fusion block.
“Width” indicates the channel dimensions of feature maps.''')

st.write('''
 We can see that by flexibly controlling the number of channels in different
scales, we can achieve much higher accuracy than sharing the same channels at all scales.
Best performance is obtained when depth equals 3 and width equals (96, 192, 384).
2) GFPN enhances feature interaction by queen-fusion, but
it also brings lots of extra upsampling and downsampling
operators. 

The benefits of those upsampling and downsampling operators are compared and results are shown in
Table.3.
''')

fig = go.Figure(data=[go.Table(
    header=dict(values=[' upsampling', 'downsampling','Latency','FLOPs','AP'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['','✓','','✓'], # 1st column
                       ['','','✓','✓'],
                       [ 3.62, 4.19, 3.83,4.58],
                       [33.3, 37.7, 37.8,  42.8],
                       [44.2, 44.5, 45.6,  45.9]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)
st.caption('Table 3. Ablation Study on the connection of queen-fusion')

st.write('''
We can see that the additional upsampling operator
results in a latency increase of 0.6ms, while the accuracy
improvement was only 0.3mAP, far less than the benefit
of the additional downsampling operator. Therefore, under
the constraints of real-time detection, we remove the extra
upsampling operation in queen-fusion. 3) In the feature
fusion block, we first replace original 3x3-convolutionbased feature fusion with CSPNet and obtain 4.2 mAP
gain. Afterward, we upgrade CSPNet by incorporating reparameterization mechanism and connections of efficient
layer aggregation networks (ELAN) [33]. Without bringing
extra huge computation burden, we achieve much higher
precision. The results of comparison are listed in Table.4.
''')

fig = go.Figure(data=[go.Table(
    header=dict(values=[' Merge−Style', 'Latency','FLOPs','AP'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['Conv','CSP','CSP + Reparam','CSP + Reparam + ELAN'], # 1st column
                       [ 3.64, 3.72, 3.72, 3.83],
                       [ 44.3, 36.7, 36.7,  37.8],
                       [40.2, 44.4, 45.0,  45.6]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)
st.caption('''
Table 4. Ablation study on the feature fusion style. CSP
denotes the Cross-Stage-Partial Connection. Reparam [5, 6]
denotes applying re-parameter mechanism on the bottleneck of
CSP. ELAN denotes the connections of efficient layer aggregation
networks.

''')

st.write('**ZeroHead and AlignOTA**')

st.write('''
In recent advancements of object detection, decoupled head is widely used [9, 17, 37]. With the decoupled head,
those models achieve higher AP, while the latency grows significantly. To trade off the latency and the performance,
we have conducted a series of experiments to balance the importance of neck and head, and the results are shown in
Table.5
''')

fig = go.Figure(data=[go.Table(
    header=dict(values=[' Neck(width/depth)', 'Head(width/depth)',' Latency(ms)','AP'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['(1.0/1.0)','(1.0/0.50)','(1.0/0.33)','(1.0/0.0)'], # 1st column
                       [  '(1.0/0.0)', '(1.0/1.0)',  '(1.0/2.0)', '(1.0/3.0)'],
                       [  3.83, 3.79,  3.85,  3.87],
                       [45.6,  44.9,  43.7,   41.2]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)
st.caption('Table 5. Studies on the balance between RepGFPN and ZeroHead.')

st.write('''
From the experiments, we find that “large neck, small head” would lead to better performance. Hence, we
discard the decoupled head in previous works [9, 17, 37],but only left a task projection layer, i.e., one linear layer for
classification and one linear layer for regression. We named our head as ZeroHead as there is no other training layers in
our head. ZeroHead can save computations for the heavy RepGFPN neck to the greatest extent. It is worth noticing
that ZeroHead essentially can be considered as a coupled head, which is quite a difference from the decoupled heads
in other works [9, 17, 31, 37]. In the loss after head, following GFocal [19], we use Quality Focal Loss (QFL)
for classification supervision, and Distribution Focal Loss (DFL) and GIOU loss for regression supervision. QFL
encourages to learn a joint representation of classification and localization quality. DFL provides more informative
and precise bounding box estimations by modeling their locations as General distributions.

The training loss of the proposed DAMO-YOLO is formulated as:''')

st.latex(r'''Loss = \alpha Loss_{QFL} + \beta Loss_{DFL} + \gamma Loss_{GIOU}''')

st.write(
    '''
    Besides head and loss, label assignment is a crucial component during detector training, which is responsible
for assigning classification and regression targets to predefined anchors. Recently, dynamic label assignment such
as OTA [8] and TOOD [7] is widely acclaimed and achieves significant improvements compares to static one [41]. Dynamic label assignment methods assign labels according to
the assignment cost between prediction and ground truth,e.g., OTA [8]. Although the alignment of classification and
regression in loss is widely studied [7, 19], the alignment between classification and regression in label assignment is
rarely mentioned in current works. The misalignment of classification and regression is a common issue in static
assignment methods [41]. Though dynamic assignment alleviates the problem, it still exists due to the unbalance
of classification and regression losses, e.g., CrossEntropy and IoU Loss [39]. To solve this problem, we introduce the
focal loss [21] into the classification cost, and use the IoU of prediction and ground truth box as the soft label, which
is formulated as follows:
    '''
)

st.latex(r'''AssignCost =  C_{reg} +  C_{cls}''')
st.latex(r'''\alpha =  IoU(reg_{gt}, reg_{pred})''')
st.latex(r'''C_{reg} =  -\ln(\alpha)''')
st.latex(r'''C_{cls} =  (\alpha - cls_{pred})^2 \times CE(cls_{pred},\alpha) ''')

st.write(
    '''
    With this formulation, we are able to choose the classification and regression aligned samples for each target.
    Besides the aligned assignment cost, following OTA [8], we form the solution of aligned assignment cost from a global
    perspective. We name our label assignment as AlignOTA.The comparison of label assignment methods is conducted
    in Table.6. We can see that AlignOTA outperforms all other label assignment methods
    '''
)

fig = go.Figure(data=[go.Table(
    header=dict(values=[' Assigner', 'AP'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['ATSS [41]','simOTA [8]','TOOD [7]','AlignOTA'], # 1st column
                       [43.1,  44.2,  45.4, 45.6]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)
st.caption('Table 6. The comparison of different on MSCOCO val dataset')

st.write('**Distillation Enhancement**')

st.write('''
Knowledge Distillation (KD) [13] is an effective method to further boost the performance of pocket-size models.
Nevertheless, applying KD on YOLO series sometimes can not achieve significant improvements as hyperparameters
are hard to optimize and features carry too much noise.In DAMO-YOLO, we first make distillation great again
on models of all sizes, especially on the small size. We adopt the feature-based distillation to transfer dark knowledge, which can distill both recognition and localization
information in the intermediate feature maps [14]. We conduct fast validation experiments to choose a suitable
distillation method for our DAMO-YOLO. The results are shown in Table.7
''')

fig = go.Figure(data=[go.Table(
    header=dict(values=[' Methods', 'Epochs', 'AP'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[['Mimicking [18]','MGD [38]','CWD [27]'],
                       [36,36,36], # 1st column
                       [ 40.2,  39.6,  40.7]],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=400)
st.plotly_chart(fig,use_container_width=True)
st.caption('Table 7. Studies on the distillation methods for DAMO-YOLO on MSCOCO val dataset. The baseline of student is 38.2.')

st.write('''We conclude that CWD is more fit for our models, while MGD is worse than Mimicking as complex
hyperparameters make it not general enough.Our proposed distillation strategy is split into two stages:

1) Our teacher distills the student at the first stage (284 epochs) on strong mosaic domain. Facing the challenging 
    augmented data distribution, the student can further extract information smoothly under the teacher’s guidance.

2) The student finetunes itself on no mosaic domain at the second stage (16 epochs). The reason why we do not adopt
    distillation at this stage is that, in such a short period, the teacher’s experience will damage the student’s performance
    when he wants to pull the student in a strange domain (i.e.,no mosaic domain). A long-term distillation would weaken
    the damage but is expensive. So we choose a trade-off to make the student independent.

In DAMO-YOLO, the distillation is equipped with two advanced enhancements:

1) Align Module. On the one hand, it is a linear projection layer to adapt student feature’s to the same resolution
    (C, H, W) as teacher’s. On the other hand, forcing the student to approximate teacher feature directly leads to
    minor gains compared to the adaptive imitation [36]. 

2) Channel-wise Dynamic Temperature. Inspired by PKD [2], we add a normalization to teacher and student features, to
    weaken the effect the difference of real values brings. After subtracting the mean, standard deviation of each channel
    would function as temperature coefficient in KL loss. 
    
Besides, we present two key observations for a better usage of distillation. One is the balance between distillation
and task loss.
''')
st.image('image/rs-3.png')
st.caption('''
Figure 3. The classification loss and AP curves of distillation. The
distillation loss weight is set to 0.5, 2, and 10 respectively. The
classification loss has a significantly fast convergence with higher
accuracy when the distillation loss weight is set to 0.5.
''')
st.write(
    '''
    As shown in Fig.3, when we focus more on distillation (weight=10), the classification loss in student
has a slow convergence, which results in a negative effect.The small loss weight (weight=0.5) hence is necessary
to strike a balance between distillation and classification.The other is the shallow head of detector. We found that
the proper reduction of the head depth is beneficial to the feature distillation on neck. The reason is that when the gap
between the final outputs and the distilled feature map is closer, distillation can have a better impact on decision
    '''
)
st.header('Implementation Details')
st.write('''
    Our models are trained 300 epochs with SGD optimizer.The weight decay and SGD momentum are 5e-4 and 0.9
respectively. The initial learning rate is 0.4 with a batch size of 256, and the learning rate decays according to a cosine
schedule. Following YOLO-Series [9, 17, 31, 33] model exponential moving average (EMA) and grouped weight
decay are used. To enhance data diversity, Mosaic [1,31] and Mixup [40] augmentation is a common practice.
However, recent advancement [3, 42] shows that properly designed box-level augmentation is crucial in object detection. Inspired by this, we apply Mosaic and Mixup
for image-level augmentation and employ the box-level augmentation of SADA [3] after image-level augmentation
for more robust augmentation.
    ''')

st.header('Comparison with the SOTA')
st.write('''
The final performance compared with SOTAs is listed
in Table.8. For a comprehensive look, we list the results
with and without distillation. It shows that our DAMOYOLO family outperforms all YOLO series in accuracy and
speed, which indicates that our method can detect objects
effectively and efficiently
''')
st.image('image/table-1.png')
st.caption('''Table 8. Comparison with the state-of-the-art single-model detectors on MSCOCO test-dev. * denotes using distillation. Latency is tested
by ourself on T4 GPUs, while other results are from the corresponding papers. FPS is reported based on TensorRT engine in FP16.''')


st.header('Conclusions')
st.info('''
    In this paper, we propose a new object detection method called DAMO-YOLO, the performance of which is superior to other methods in YOLO series.Its advantages come from new techs, including
 6 MAE-NAS backbone, efficient RepGFPN neck,ZeroHead, AlignedOTA label assignment and distillation enhancement.
   ''')

st.header('Reference Link')
st.write("A Report on Real-Time Object Detection Design - https://arxiv.org/pdf/2211.15444v2.pdf")
st.write('Repository link - https://github.com/tinyvision/damo-yolo')

st.header('References')

with st.expander("Expand"):
    references = read_markdown_file("pages/reference.md")
    st.write(references, unsafe_allow_html=True)
    
