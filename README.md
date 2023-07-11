# VGG-optiVMD: An powerful algorithm to improve SER

# Author: David Hason Rudd

# University of Technology Sydney

Please cite the paper (below DOI) if you are interested in the proposed methodological approach 

DOI: 10.1007/978-3-031-33380-4_17

Emotion recognition (ER) from speech signals is a robust approach since it cannot be imitated like facial expression or text based sentiment analysis. Valuable information underlying the emotions are significant for human-computer interactions enabling intelligent machines to interact with sensitivity in the real world. Previous ER studies through speech signal processing have focused exclusively on associations between different signal mode decomposition methods and hidden informative features. However, improper decomposition parameter selections lead to informative signal component losses due to mode duplicating and mixing. In contrast, the current study proposes VGG-optiVMD, an empowered variational mode decomposition algorithm, to distinguish meaningful speech features and automatically select the number of decomposed modes and optimum balancing parameter for the data fidelity constraint by assessing their effects on the VGG16 flattening output layer. Various feature vectors were employed to train the VGG16 network on different databases and assess VGG-optiVMD reproducibility and reliability. One, two, and three-dimensional feature vectors were constructed by concatenating Mel-frequency cepstral coefficients, Chromagram, Mel spectrograms, Tonnetz diagrams, and spectral centroids. Results confirmed a synergistic relationship between the fine-tuning of the signal sample rate and decomposition parameters with classification accuracy, achieving state-of-the-art 96.09\% accuracy in predicting seven emotions on the Berlin EMO-DB database.


Modeling: 
The aim of modeling was to enhance informative data within the feature vectors and avoid overfitting. Augmentation effects on classification accuracy were assessed using diverse K and alpha sets.

Conclusion:
The findings provide solid empirical confirmation of the key role of the sampling rate, the number of the decomposed mode, K and the balancing parameter of the data-fidelity constraint, alpha, in the performance of the emotion classifier. Taken together, these findings suggest that VMD decomposition parameters K (2-6) and alpha (2000-6000) are optimum values on both the RAVDESS and EMODB databases. 
The proposed VGG-optiVMD algorithm improved the emotion classification to a state-of-the-art result with a test accuracy of 96.09% in the Berlin EMO-DB and 86.21% in the RAVDESS datasets. 
