# Face Recognition Project

This project implements a face recognition system using a custom-trained Siamese Model with the MobileNet architecture. The model is trained on the Labeled Faces in the Wild (LFW) dataset.

## Table of Contents
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Pretrained MobileNet](#pretrained-mobilenet)
- [Usage](#usage)
- [References](#references)

## Dataset

### Downloading the Dataset
The dataset for custom training can be downloaded from the LFW website:

- [Download Dataset](https://vis-www.cs.umass.edu/lfw/)
  
Please download the dataset in the 'All images as gzipped tar file' format. After downloading, extract the contents into the `data/negative` folder for retraining the model.

## Model Training

### Training Notebook
The object detection model used in this project is a trained Siamese Model. The training process is documented in a Jupyter notebook, which can be accessed here:

- [Download Training Notebook](https://huggingface.co/Changchoichang2104/siamesemodel_facerecognition/blob/main/face-recognition_final.ipynb)

### Training Process
The training process is highly influenced by the following YouTube playlist:

- [Face Recognition Training Tutorial](https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH)

## Pretrained MobileNet

The MobileNet architecture is used as the base model, with additional layers to fit the Siamese model structure. The MobileNet used is within the TensorFlow framework.

## Usage

## Usage

Follow these steps to set up and use the face recognition system:

1. **Train Your Model**:
   - Use the provided Jupyter notebook to train your model. Make sure you have downloaded and prepared the dataset as described in the [Dataset](#dataset) section.

2. **Set Up the Project**:
   - Clone this repository to your local machine.
   - Place your trained model within the repository's directory.

3. **Set Up the `application_data` Folder**:
   - Create an `application_data` folder within the project directory.
   - Inside `application_data`, create a `verification_image` folder. This should contain the positive image, which you can copy from the training process.
   - Also, create an `input_image` folder within `application_data`. This is where you'll place the input images for the application to process.

4. **Run the Application**:
   - Execute `app.py` to start the application with your trained model. Ensure that the `application_data` folder is set up as described above.

## References

- Labeled Faces in the Wild (LFW): [Visit LFW Website](https://vis-www.cs.umass.edu/lfw/)
- Siamese Network Tutorial: [YouTube Playlist](https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH)
- Hugging Face Model Repository: [Siamese Model on Hugging Face](https://huggingface.co/Changchoichang2104/siamesemodel_facerecognition)

## Have Questions or Feedback?

If you have any questions, concerns, or feedback about this project, we'd love to hear from you! Feel free to open an issue in this repository or contact the contributors directly. Your input is valuable to us and will help in improving this project.

---

Your contributions and feedback make this project better for everyone. Thank you for being a part of our community!
