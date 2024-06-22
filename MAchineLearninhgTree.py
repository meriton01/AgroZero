import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn_porter import Porter

# Load data from CSV
data = pd.read_csv('sensor_data.csv')

# Separate features (X) and target (y)
X = data[['TDS', 'pH', 'Soil_Moisture']]
y = data['Condition']  # 0 for good, 1 for bad

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Export the trained model to C header file
porter = Porter(model, language='c')
output = porter.export(embed_data=True)

# Save the exported model as a C header file
with open('model.h', 'w') as f:
    f.write(output)

