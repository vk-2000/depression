import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns
import pickle

depressed_tweets = pd.read_csv("clean_d_tweets.csv")
non_depressed_tweets = pd.read_csv("clean_non_d_tweets.csv")

depressed_tweets.dropna(subset=['tweet'], inplace=True)
non_depressed_tweets.dropna(subset=['tweet'], inplace=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(depressed_tweets['tweet'].values.tolist() + non_depressed_tweets['tweet'].values.tolist())
y = [1]*len(depressed_tweets) + [0]*len(non_depressed_tweets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(kernel='linear', probability=True),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

best_model = None
best_accuracy = 0

results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:  # for SVC
        y_prob = model.decision_function(X_test)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # scaling to [0, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    class_distribution = {'Depressed': len(depressed_tweets), 'Non-Depressed': len(non_depressed_tweets)}

    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Classification Report": class_report,
        "ROC Curve": (fpr, tpr, roc_auc),
        "Confusion Matrix": conf_matrix,
        "Class Distribution": class_distribution
    }
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("Model Performance Metrics:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        if metric_name not in ['ROC Curve', 'Confusion Matrix', 'Class Distribution']:
            print(f"{metric_name}: {value}")

accuracies = {model_name: metrics['Accuracy'] for model_name, metrics in results.items()}
print("\nAccuracy Comparison:")
for model_name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name}: {accuracy}")

for model_name, metrics in results.items():
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    fpr, tpr, roc_auc = metrics['ROC Curve']
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")

    plt.subplot(1, 3, 2)
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')

    plt.subplot(1, 3, 3)
    class_distribution = metrics['Class Distribution']
    plt.bar(class_distribution.keys(), class_distribution.values(), color=['blue', 'green'])
    plt.xlabel('Class')
    plt.ylabel('Number of Tweets')
    plt.title(f'{model_name} Class Distribution')

    plt.suptitle(f'Model: {model_name}')

    plt.tight_layout()
    plt.show()

with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
