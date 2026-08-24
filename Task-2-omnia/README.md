# 🌸 Iris Flower Classification using KNN

## 📌 Project Overview

This project is a Machine Learning classification project using the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

The model predicts the species of an Iris flower based on four measurements:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The project demonstrates a complete Machine Learning workflow from data preparation to model evaluation and prediction.

---

## 🌸 Dataset

The Iris dataset is provided by Scikit-learn.

- **Number of samples:** 150
- **Number of features:** 4
- **Number of classes:** 3

### Target Classes

- Setosa
- Versicolor
- Virginica

---

## ⚙️ Machine Learning Workflow

The project follows these steps:

1. Load the Iris dataset
2. Explore the dataset
3. Separate features and target
4. Split the data into training and testing sets
5. Apply feature scaling using StandardScaler
6. Test different K values
7. Select the best K
8. Train the final KNN model
9. Make predictions
10. Evaluate the model
11. Visualize the results
12. Create an interactive prediction demo

---

## 🤖 Model

The classification algorithm used in this project is:

**K-Nearest Neighbors (KNN)**

Different values of K from 1 to 15 were tested to find the best-performing value.

### Best Parameters

- **Best K:** 1
- **Test Accuracy:** 96.67%

---

## 📊 Model Evaluation

The final model achieved an accuracy of:

**96.67%**

The model correctly classified **29 out of 30** test samples.

### Confusion Matrix

```text
[[10  0  0]
 [ 0 10  0]
 [ 0  1  9]]
```

This shows that:

- All Setosa samples were classified correctly.
- All Versicolor samples were classified correctly.
- 9 out of 10 Virginica samples were classified correctly.
- 1 Virginica sample was classified as Versicolor.

---

## 📈 Visualizations

The project includes two visualizations:

### 1. KNN Accuracy for Different K Values

Shows the model accuracy for different values of K and helps identify the best K.

### 2. Confusion Matrix

Shows the correct and incorrect predictions for each Iris class.

Both visualizations are available in the `Visualization` folder.

---

## 🖥️ Interactive GUI Demo

The project includes a graphical interface that allows the user to enter the four flower measurements and get a prediction from the trained KNN classifier.

The GUI displays:

- Best K
- Test Accuracy
- Flower measurements input
- Predicted flower species

### Example

Input:

```text
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
```

Prediction:

```text
SETOSA
```

---

## 🎬 Demo

Watch the project demonstration:

**YouTube Demo:** <https://youtu.be/hwHDhDRqR4c>

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- Matplotlib
- Tkinter

---

## 📁 Project Structure

```text
Task-2-omnia/
│
├── task2.py
├── iris_gui.py
├── README.md
│
└── Visualization/
    ├── figure1_confusion_matrix.png
    └── figure2_knn_accuracy.png
```

---

## ▶️ How to Run

Install the required libraries:

```bash
pip install scikit-learn matplotlib
```

Run the Machine Learning project:

```bash
python task2.py
```

Run the GUI demo:

```bash
python iris_gui.py
```

---

## 🎯 Conclusion

This project demonstrates a complete Machine Learning classification workflow using the Iris dataset and K-Nearest Neighbors.

The final model achieved **96.67% accuracy** on the test set and was integrated into an interactive GUI for flower classification.

---

## 👩‍💻 Author

**Omnia Ayman**