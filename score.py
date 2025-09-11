from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# True labels and Predicted labels
# Based on the confusion matrix [[4,1],[1,4]]
y_true = ['Venomous', 'Venomous', 'Venomous', 'Venomous', 'Non-venomous',
          'Non-venomous', 'Non-venomous', 'Non-venomous', 'Non-venomous', 'Venomous']

y_pred = ['Venomous', 'Venomous', 'Venomous', 'Venomous', 'Non-venomous',
          'Venomous', 'Non-venomous', 'Non-venomous', 'Non-venomous', 'Non-venomous']

# Convert labels to binary for calculation purposes
label_map = {'Venomous': 1, 'Non-venomous': 0}
y_true_binary = [label_map[label] for label in y_true]
y_pred_binary = [label_map[label] for label in y_pred]

# Calculate metrics
accuracy = accuracy_score(y_true_binary, y_pred_binary)
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)
f1 = f1_score(y_true_binary, y_pred_binary)

# Print results
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")