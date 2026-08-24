# ==========================================
# Iris Flower Classification using KNN
# ==========================================


# ==========================================
# 1. Load the Iris Dataset
# ==========================================

from sklearn.datasets import load_iris

iris = load_iris()

# Display basic information about the dataset
print("Number of samples:", iris.data.shape[0])
print("Number of features:", iris.data.shape[1])
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = iris.data
y = iris.target

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# 3. Split the Dataset into Training and Testing
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 4. Feature Scaling
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining data scaled successfully.")
print("Testing data scaled successfully.")


# ==========================================
# 5. Find the Best K Value
# ==========================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

k_values = range(1, 16)
accuracies = []

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy)


# Find the K value with the highest accuracy
best_k = k_values[accuracies.index(max(accuracies))]

print("\nBest K:", best_k)
print("Best Accuracy:", max(accuracies))


# ==========================================
# 6. Visualize Accuracy for Different K Values
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(k_values, accuracies, marker="o")

plt.title("KNN Accuracy for Different K Values")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.xticks(list(k_values))
plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================
# 7. Train the Final KNN Model
# ==========================================

# Use the best K found above
knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_scaled, y_train)

print("\nFinal KNN model trained successfully.")


# ==========================================
# 8. Make Predictions on Test Data
# ==========================================

y_pred = knn.predict(X_test_scaled)

print("\nPredictions:")
print(y_pred)


# ==========================================
# 9. Evaluate the Final Model
# ==========================================

from sklearn.metrics import classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Model Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 10. Visualize the Confusion Matrix
# ==========================================

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=iris.target_names,
    cmap="Blues"
)

plt.title("Final KNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# ==========================================
# 11. Interactive Iris Flower Prediction Demo
# ==========================================

print("\n==========================================")
print("🌸 Iris Flower Prediction Demo")
print("==========================================")

sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))


# Create a new flower sample
new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]


# Scale the new input using the same scaler
new_flower_scaled = scaler.transform(new_flower)


# Predict the flower class
prediction = knn.predict(new_flower_scaled)


# Display the prediction
print("\n🌸 Predicted Flower:", iris.target_names[prediction[0]])