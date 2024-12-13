# ExploreIndonesia

This project uses a deep learning model for recognizing handwritten characters using a convolutional neural network (CNN).

## Dataset
The dataset used in this project is the **A-Z Handwritten Dataset**, which can be downloaded from the following links:

- [Kaggle Dataset Link](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) <!-- Link ke Kaggle -->

# Enhanced OCR Model
This repository contains the code for an enhanced OCR model using a Convolutional Neural Network (CNN). The model is designed to recognize both handwritten digits and letters with high accuracy, using data augmentation and effective callbacks to improve performance.

## Project Structure
The project is organized as follows:
- *OCR.ipynb* Contains the Python code for the OCR model.

## Requirements
To run the code, you will need the following libraries installed:
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- JSON
- Scikit-learn

## Usage
1. *Install the required libraries:*
   ```bash
   pip install tensorflow keras numpy pandas opencv-python matplotlib seaborn json scikit-learn

2. Download the dataset:
* The dataset used in this project is the "A-Z Handwritten Data.csv" file. You can download it from - [Kaggle Dataset Link](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) <!-- Link ke Kaggle -->
* Place the dataset in the same directory as the Python script.

3. Run the Python script:
* Open the "Enhanced OCR Model.docx" file in a Python IDE or text editor.
* Execute the code.

Model Training
* The script loads the A-Z and MNIST datasets.
* It preprocesses the data and splits it into training and validation sets.
* The model is trained using data augmentation and callbacks to improve performance.
* The training history is plotted to visualize the model's accuracy and loss over epochs.

Model Evaluation
* The model is evaluated on the validation dataset to check its performance.
* The classification report and confusion matrix are printed to assess the model's accuracy and identify areas for improvement.

Model Saving and Loading
* The trained model is saved to a file for later use.
* The script includes code to load a previously saved model.

Prediction
* The script demonstrates how to use the trained model to predict characters in an image.
* It extracts characters using image processing techniques and displays the predicted text along with confidence scores and bounding box positions.

Contributing
Contributions are welcome! If you have any suggestions or improvements, please feel free to submit a pull request.

License

This project is licensed under the MIT License.
*To download this file:*

1. *Right-click* on the text above.
2. *Select "Save as"* or "Save link as".
3. *Choose a location* to save the file.
4. *Name the file "README.md"* (without the quotes).
5. *Click "Save"*. 

You now have a README.md file that you can use for your OCR project.
