# CITL-AI
The training code of paper **Improving the accuracy and efficiency of cervical cytology screening with cytologist-in-the-loop artificial intelligence: a cross-sectional, multicentre, proof-of-concept study**. 

The AI system used in this study is a two-stage method based on deep learning algorithms. Firstly, a abnormal cell detector is trained with labeled images to detect abnormal cells in the cervical cytological slide, and the detected abnormal cells in the slide are selected to form a slide bag for slide classification. Then, a slide classifier is trained with the labeled slides for the whole slide recognition. 

## Stage1 - Abnormal cell detector
Please go to [stage1](stage1).

## Stage2 - Slide classifier
Please go to [stage2](stage2).
