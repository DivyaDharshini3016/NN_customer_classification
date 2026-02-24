# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="672" height="614" alt="Screenshot 2026-02-24 210046" src="https://github.com/user-attachments/assets/b6782b9b-8b2a-4ceb-9482-45f09ca83836" />


## DESIGN STEPS

STEP 1: Import necessary libraries and load the dataset.

STEP 2:
Encode categorical variables and normalize numerical features.

STEP 3:
Split the dataset into training and testing subsets.

STEP 4:
Design a multi-layer neural network with appropriate activation functions.

STEP 5:
Train the model using an optimizer and loss function.

STEP 6:
Evaluate the model and generate a confusion matrix.

STEP 7:
Use the trained model to classify new data samples.

STEP 8:
Display the confusion matrix, classification report, and predictions.

## PROGRAM

### Name: Divya Dharshini S
### Register Number: 212224240039

```
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
```

```
 # Initialize model
model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

```
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

  if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

## Dataset Information

<img width="1097" height="631" alt="image" src="https://github.com/user-attachments/assets/3a5d5850-6e66-474f-8e64-1988b865e7f3" />

## OUTPUT

### Confusion Matrix

<img width="531" height="447" alt="image" src="https://github.com/user-attachments/assets/d2db08d3-2238-41b4-ad36-02ef7764de58" />


### Classification Report

<img width="635" height="365" alt="image" src="https://github.com/user-attachments/assets/a4dac700-67d4-4e16-9c4c-128b2b93c149" />



### New Sample Data Prediction

<img width="1277" height="305" alt="image" src="https://github.com/user-attachments/assets/89b6105b-ed07-47e8-98db-c34ea87c53c9" />


## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
