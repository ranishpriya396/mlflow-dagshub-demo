import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# Initialize DagsHub tracking
mlflow.set_tracking_uri("https://dagshub.com/ranishpriya396/mlflow-dagshub-demo.mlflow")
dagshub.init(repo_owner='ranishpriya396', repo_name='mlflow-dagshub-demo', mlflow=True)

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Hyperparameters
max_depth = 10
n_estimators = 4
with mlflow.start_run():
    # Train
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators = n_estimators)
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics & corrected parameter names
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('n_estimatirs', n_estimators)  # Fixed parameter name
    mlflow.log_param('max_depth', max_depth)
    
    # Metadata tags
    mlflow.set_tag('author', 'ranish')        # Fixed typo
    mlflow.set_tag('model', 'random_forest')

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))     # Using explicit subplot objects is safer
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')

    # FIXED ORDER: Save -> Close -> Log
    plot_path = "confusion_matrix.png"
    fig.savefig(plot_path, bbox_inches='tight') 
    plt.close(fig)                              
    mlflow.log_artifact(plot_path)              

    # Log model
    mlflow.sklearn.log_model(rf , 'randomforest')

print('Accuracy:', accuracy)
