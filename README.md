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

1. **Train Your Model**: Use the Jupyter notebook to train your model. Ensure you have downloaded and prepared the dataset as described above.
2. **Set Up the Project**: Clone this repository and place your trained model within the directory.
3. **Run the Application**: Execute `app.py` to start the application with your trained model.

## References

- Labeled Faces in the Wild (LFW): [Visit LFW Website](https://vis-www.cs.umass.edu/lfw/)
- Siamese Network Tutorial: [YouTube Playlist](https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH)
- Hugging Face Model Repository: [Siamese Model on Hugging Face](https://huggingface.co/Changchoichang2104/siamesemodel_facerecognition)

## Have Questions or Feedback?

If you have any questions, concerns, or feedback about this project, we'd love to hear from you! Feel free to open an issue in this repository or contact the contributors directly. Your input is valuable to us and will help in improving this project.

---

Your contributions and feedback make this project better for everyone. Thank you for being a part of our community!
