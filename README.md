DETECTION OF BREAST CANCER USING MACHINE LEARNING , DEEP LEARNING AND NEURAL NETWORKING

1.INTRODUCTION: 
Breast cancer is one of the most common and deadly cancers among women worldwid
 e. Early detection and accurate diagnosis are important for optimal treatment and bette
 r patient outcomes. But traditional diagnostic methods, such as breast examination and
 traditional screening methods, often face issues with sensitivity and specificity, which c
 an make the test invisible or negative. 
 
 2.LITERATURE SURVEY: 
In recent years, the application of deep learning in medical imaging has made significan
 t progress, with convolutional neural networks (CNN) at the forefront. CNNs are designe
 d to achieve and transform learning the spatial hierarchy of features from input images, 
showing great potential in many image analysis applications, including detection, segm
 entation, and distribution. Esteva et al. (2017) highlighted the ability of deep learning alg
 orithms using CNNs to match or exceed human performance in skin cancer classificati
 on, setting a precedent for similar applications in other fields (such as mammography f
 or breast cancer diagnosis). 
 3. METHODS: 
3.1 CNN: 
Convolutional neural networks (CNN) are a class of deep learning algorithms that are ve
 ry successful in computer vision and image analysis. They aim to achieve and adapt lear
 ning of the spatial hierarchy of features of the input image using layers, layers, and overl
 ays. Convolutional layers apply a series of filters to the input image to create feature ma
 ps that capture local patterns such as edges, textures, and shapes. Dynamic processin
 g such as linear regression (ReLU) allows one to examine complex patterns by showing 
disparities in the patterns. 
3.2 Inception V3 Model: 
The Inception V3 model has gained attention for its performance and efficiency in image 
recognition tasks. The first version of the Inception architecture was introduced in the 
paper "Rethinking the Inception Architecture for Computer Vision" by Szegedy et al. They 
did it in 2016 Several innovations that improve both the depth and width of the network 
are introduced in the model.
4. Materials: 
4.1 Experimental Datasets for Model Training: 
The Digital Database for Screening Mammography Selected Breast image Subset is a 
database of mammography images designed to support research on breast cancer 
screening and diagnosis. The original DDSM dataset is one of the first publicly available 
mammogram datasets. Valuable resources for training and testing machine learning 
models are in this database. There are 199 events. The total number of images included 
in each case was 10,239. 
4.2  Experimental Datasets for Model Testing: 
5.000000 
The Mini-DDSM (Mini Digital Database for Screening Mammography) is a 
carefully selected subset of the larger DDSM dataset, designed to provide a 
manageable yet representative sample of mammographic images for 
research and testing purposes. The Mini-DDSM dataset retains the essential 
characteristics and diversity of the full DDSM dataset, making it an excellent 
resource for evaluating the performance of machine learning models, 
particularly those focused on breast cancer detection and diagnosis.
5.Proposed Methodology: 
This research paper proposes an optimal approach and best method for the identification 
of breast cancer using the mammography images from the dataset and uses Inception 
V3 model for detection of the cancer in the given image. We use methods like data 
classification, data preprocessing, data augmentation and data visualization.
5.1 Pre-Processing : 
During image preprocessing, raw image data undergoes manipulation to convert it into a 
format that is both useful and meaningful. This process enables the enhancement of 
specific qualities that are crucial for computer vision applications. Preprocessing serves 
as the initial stage in preparing image data. Various techniques are utilized in image 
preprocessing to ensure the proper functioning of machine learning algorithms, such as 
resizing images to a standardized dimension. OpenCV offers a method for resizing 
images.
5.2 Data Augmentation: 
Data augmentation is a statistical technique which allows maximum likelihood 
estimation from incomplete data. Data augmentation has important applications in 
Bayesian analysis, and the technique is widely used in machine learning to reduce 
overfitting when training machine learning models, achieved by training models on 
several slightly-modified copies of existing data.
5.3 Experiment Setup for CNN Model: 
The experiment begins with the preparation of datasets. The training data is sourced from 
the CBIS-DDSM dataset, which includes 277,524 images. For testing, the Mini-DDSM 
dataset, comprising 880 images, is used. Preprocessing steps include normalizing pixel 
values to the range [0, 1], resizing images to a fixed size compatible with the CNN model 
(e.g., 299x299 for Inception V3), and applying data augmentation techniques. These 
augmentation techniques, such as random rotations, flips, zooms, and shifts, are 
employed to enhance model generalization and robustness by simulating various real
world scenarios.
5.4 Experiment Setup for Inception V3 Model: 
The experiment starts with the preparation of the datasets. The CBIS-DDSM dataset, 
containing 277,524 images, is used for training, while the Mini-DDSM dataset, with 880 
images, is utilized for testing. Preprocessing involves normalizing pixel values to the 
range [0, 1] and resizing images to 299x299 pixels to match the input size required by the 
Inception V3 model. Data augmentation techniques, such as random rotations, flips, 
zooms, and shifts, are applied to increase the diversity of the training data and help the 
model generalize better to new, unseen images. 
The Inception V3 model, pre-trained on the ImageNet dataset, is chosen as the base 
model. This pre-trained model benefits from transfer learning, which leverages features 
learned from a large, diverse dataset. 
6.1 Dataset: 
CBISDDSM (Selective Body Imaging of DDSM) is an updated and standardized version o
 f the Digital Database of Screening Mammography (DDSM). DDSM is a repository of 2,62
 0 studies related to mammography imaging. There are normal, benign, and malignant c
 onditions as well as evidence of pathology. Large amounts of data and ground truth reco
 gnition make DDSM an important tool for the development and evaluation of decision 
upport. The CBISDDSM collection includes DDSM techniques selected and curated by 
mammographers. Images are decompressed and converted to DICOM format. It also in
 cludes modified ROI segmentations and bounding boxes, as well as pathological diagno
 sis of the study material. An article detailing the use of these data is available at https://
 www.nature.com/articles/sdata2017177.
 7. Results: 
7.1. Data Visualization: 
Visualizing the preprocessing and augmentation steps helps ensure that the 
transformations applied are meaningful and will aid in generalizing the model. For 
example, displaying a sample of original mammography images alongside their 
augmented versions can showcase the variety introduced through data augmentation 
techniques such as rotation, flipping, zooming, and shifting. This step is crucial to 
confirm that the augmentations are reasonable and retain the medical relevance of the 
images. It can be done using Python libraries like Matplotlib and Keras’s Image Data 
Generator.
7.2 Performance Analysis: 
Accuracy is a primary metric used to evaluate the performance of classification models, 
representing the proportion of correctly classified instances out of the total instances. 
For both the standard CNN and Inception V3 models, accuracy can be calculated on both 
training and testing datasets. Generally, Inception V3, with its more advanced 
architecture and deeper network, is expected to achieve higher accuracy compared to a 
traditional CNN due to its ability to capture more complex features. 
Precision: Precision measures the proportion of true positive predictions among all 
positive predictions. It indicates how many of the positively predicted cases were actually 
positive. 
Precision = [(True Positive) / (True Positive + False Positive)] 
Recall: Recall measures the proportion of true positive predictions among all actual 
positive cases. It indicates how well the model identifies positive cases. 
Recall = [(True Positive) / (True Positive + False Negative)] 
F1-Score: The F1-Score is the harmonic mean of precision and recall, providing a single 
metric that balances both concerns. 
F1 – Score = 2 X [(precision x recall) / (precision + recall)] 
8. Future Scope: 
As CNN architectures such as Inception V3 continue to improve, they provide greater 
accuracy in detecting breast abnormalities from mammogram images. Future advances 
will focus on adapting the treatment model to be more specific and specific, making 
cancer diagnosis faster and more accurate. This can reduce adverse events and improve 
overall patient outcomes. Automated diagnosis and decision support: 
CNN can simplify the diagnostic process by analyzing mammograms. Future 
developments may include integrating CNN models into clinical workflows to provide 
radiologists with real-time decision-making. This will help prioritize issues, reduce 
translation time and ensure quality reading. Personalized Medicine and Risk 
Assessment.
9.Conclusion: 
The attainment of 99.8% accuracy using the Inception V3 model represents a remarkable 
advancement in mammography-based breast cancer detection. This achievement 
underscores the efficacy of deep learning and transfer learning techniques in harnessing 
complex patterns within mammography images. The high accuracy achieved not only 
highlights the robustness and sensitivity of the Inception V3 architecture but also signals 
its potential to significantly enhance early detection rates and patient outcomes in 
clinical settings.
