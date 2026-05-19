import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA


def install_dependencies():
    subprocess.run(
        ["pip", "install", "kaggle", "kagglehub", "pandas",
         "numpy", "matplotlib", "scikit-learn"],
        check=True
    )


def load_dataset():
    kagglehub.dataset_download("sameepvani/nasa-nearest-earth-objects")

    csv_path = None
    for root, dirs, files in os.walk(os.path.expanduser("~/.cache/kagglehub")):
        for file in files:
            if file.endswith(".csv") and "neo" in file:
                csv_path = os.path.join(root, file)

    if csv_path is None:
        raise FileNotFoundError("Dataset CSV not found. Check kagglehub download.")

    dataframe = pd.read_csv(csv_path)
    print(f"Dataset loaded: {dataframe.shape[0]} rows, {dataframe.shape[1]} columns")
    return dataframe


def plot_exploratory_analysis(dataframe):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("NASA Near-Earth Objects: Exploratory Data Analysis", fontsize=15, fontweight="bold")

    class_counts = dataframe["hazardous"].value_counts()
    axes[0].bar(
        ["Non-Hazardous", "Hazardous"],
        class_counts.values,
        color=["steelblue", "crimson"],
        edgecolor="black"
    )
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")
    for index, value in enumerate(class_counts.values):
        axes[0].text(index, value + 300, str(value), ha="center", fontweight="bold")

    for is_hazardous, color, label in [(True, "crimson", "Hazardous"), (False, "steelblue", "Non-Hazardous")]:
        diameter_subset = dataframe[dataframe["hazardous"] == is_hazardous]["est_diameter_min"]
        axes[1].hist(diameter_subset, bins=50, alpha=0.6, color=color, label=label)
    axes[1].set_title("Estimated Diameter Distribution")
    axes[1].set_xlabel("Min Diameter (km)")
    axes[1].set_xlim(0, 5)
    axes[1].legend()

    hazardous_objects = dataframe[dataframe["hazardous"] == True]
    safe_objects = dataframe[dataframe["hazardous"] == False].sample(2000, random_state=42)
    axes[2].scatter(
        safe_objects["miss_distance"] / 1e6,
        safe_objects["relative_velocity"],
        alpha=0.3, color="steelblue", s=5, label="Non-Hazardous"
    )
    axes[2].scatter(
        hazardous_objects["miss_distance"] / 1e6,
        hazardous_objects["relative_velocity"],
        alpha=0.5, color="crimson", s=8, label="Hazardous"
    )
    axes[2].set_title("Miss Distance vs Relative Velocity")
    axes[2].set_xlabel("Miss Distance (million km)")
    axes[2].set_ylabel("Relative Velocity (km/s)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def preprocess_data(dataframe):
    selected_features = [
        "est_diameter_min", "est_diameter_max",
        "relative_velocity", "miss_distance", "absolute_magnitude"
    ]

    input_features = dataframe[selected_features]
    target_labels = dataframe["hazardous"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        input_features, target_labels,
        test_size=0.2, random_state=42, stratify=target_labels
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features


def train_all_classifiers(X_train_scaled, X_test_scaled, y_train, y_test):
    classifier_collection = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    training_results = {}
    for classifier_name, classifier_model in classifier_collection.items():
        print(f"Training {classifier_name}...")
        classifier_model.fit(X_train_scaled, y_train)
        predictions = classifier_model.predict(X_test_scaled)
        probabilities = classifier_model.predict_proba(X_test_scaled)[:, 1]
        training_results[classifier_name] = {
            "model": classifier_model,
            "predictions": predictions,
            "probabilities": probabilities,
            "accuracy": accuracy_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }
        print(f"  Accuracy: {training_results[classifier_name]['accuracy']:.4f} | AUC: {training_results[classifier_name]['roc_auc']:.4f}")

    print("All classifiers trained successfully.")
    return training_results


def plot_confusion_matrices(training_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrices: All Classifiers", fontsize=15, fontweight="bold")

    for ax, (classifier_name, result) in zip(axes, training_results.items()):
        matrix = result["confusion_matrix"]
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(
            f"{classifier_name}\nAcc={result['accuracy']:.3f} | AUC={result['roc_auc']:.3f}",
            fontsize=10
        )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Haz", "Haz"])
        ax.set_yticklabels(["Non-Haz", "Haz"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for row in range(2):
            for col in range(2):
                ax.text(
                    col, row, f"{matrix[row, col]:,}",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black"
                )

    plt.tight_layout()
    plt.show()


def plot_roc_curves(training_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    curve_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for (classifier_name, result), color in zip(training_results.items(), curve_colors):
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, result["probabilities"])
        ax.plot(
            false_positive_rate, true_positive_rate,
            color=color, lw=2,
            label=f"{classifier_name} (AUC={result['roc_auc']:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: Space Object Hazard Classifiers", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(training_results, selected_features):
    random_forest_model = training_results["Random Forest"]["model"]
    feature_importances = pd.Series(
        random_forest_model.feature_importances_,
        index=selected_features
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feature_importances)))
    bars = ax.barh(feature_importances.index, feature_importances.values, color=bar_colors, edgecolor="black")
    ax.set_title("Random Forest: Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar, importance_value in zip(bars, feature_importances.values):
        ax.text(
            importance_value + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{importance_value:.3f}",
            va="center", fontweight="bold"
        )
    plt.tight_layout()
    plt.show()


def plot_pca_visualization(X_test_scaled, y_test, training_results):
    pca_transformer = PCA(n_components=2, random_state=42)
    X_pca_transformed = pca_transformer.fit_transform(X_test_scaled)
    random_forest_predictions = training_results["Random Forest"]["predictions"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA 2D: Ground Truth vs Random Forest Predictions", fontsize=14, fontweight="bold")

    for ax, label_array, chart_title in [
        (axes[0], y_test.values, "Ground Truth"),
        (axes[1], random_forest_predictions, "Random Forest Predictions")
    ]:
        for class_value, point_color, legend_label in [
            (0, "steelblue", "Non-Hazardous"),
            (1, "crimson", "Hazardous")
        ]:
            class_mask = label_array == class_value
            ax.scatter(
                X_pca_transformed[class_mask, 0],
                X_pca_transformed[class_mask, 1],
                c=point_color, alpha=0.3, s=4, label=legend_label
            )
        ax.set_title(chart_title)
        ax.legend(markerscale=3)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

    plt.tight_layout()
    plt.show()

    variance_pc1 = pca_transformer.explained_variance_ratio_[0]
    variance_pc2 = pca_transformer.explained_variance_ratio_[1]
    print(f"PCA Variance Explained: PC1={variance_pc1:.1%} | PC2={variance_pc2:.1%}")


def main():
    install_dependencies()

    dataframe = load_dataset()

    plot_exploratory_analysis(dataframe)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, selected_features = preprocess_data(dataframe)

    training_results = train_all_classifiers(X_train_scaled, X_test_scaled, y_train, y_test)

    plot_confusion_matrices(training_results)

    plot_roc_curves(training_results, y_test)

    plot_feature_importance(training_results, selected_features)

    plot_pca_visualization(X_test_scaled, y_test, training_results)


if __name__ == "__main__":
    main()
