import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("https://dagshub.com/ranishpriya396/mlflow-dagshub-demo.mlflow")
import dagshub
dagshub.init(repo_owner='ranishpriya396', repo_name='mlflow-dagshub-demo', mlflow=True)


# Load the iris dataset properly
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameters for the Random Forest model
max_depth = 10
criterion = 'entropy'


# 2. Name your project experiment
mlflow.set_experiment("dt-1")

# Apply MLflow tracking
with mlflow.start_run(run_name='run-2'):
    # Train the random forest model
    dt  = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion = criterion
    )
    dt.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters to MLflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('n_estimators', criterion)
    mlflow.log_param('max_depth', max_depth)

    # Generate, save, and log the confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the figure locally and log it as an MLflow artifact
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(__file__)

#     # Log the trained model artifact directly
    mlflow.sklearn.log_model(dt, "random_forest_model")
    mlflow.set_tag('auther', 'ranish')
    mlflow.set_tag('model', 'decision_tree')
print('accuracy:', accuracy)
