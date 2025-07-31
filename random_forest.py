import numpy as np
import os
import random
import processpcd as pr
import pandas as pd
import open3d as o3d
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def get_rand_idx(n, limit):

    idxList = []

    while len(idxList) < n:
        idx = random.randint(0, limit)
        if not idx in idxList:
            idxList.append(idx)

    return idxList

# Assume 'features' is a NumPy array of extracted point cloud features
# and 'labels' is a NumPy array of corresponding class labels
# Example: features = np.random.rand(100, 5)  # 100 points, 5 features each
#          labels = np.random.randint(0, 3, 100) # 3 classes
def rf():

      groundtruth = pr.get_groundtruth()
      idxList = get_rand_idx(3, len(groundtruth))

      fileList = [groundtruth['sequenceName'][idx] for idx in idxList]

      con_matrices = []

      for file in fileList:
            #labels = groundtruth[groundtruth['sequenceName'] == file]['classname']
            #print(f"Labels: {labels}")
            pcd, annoted, labels = pr.load_specific(file, True)

            print(f"pcd: {pcd.points[0]}\n\n"
                  f"annotated: {annoted[0]}\n\n"
                  f"labels: {labels[0]}")

            #example_pcd = o3d.geometry.PointCloud()

            #example_pcd.points = o3d.utility.Vector3dVector(annoted)

            #o3d.visualization.draw_geometries([pcd, example_pcd])

            #o3d.visualization.draw_geometries([example_pcd])

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(pcd.points, labels, test_size=0.2, random_state=42)

            # Initialize and train the Random Forest Classifier
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = rf_model.predict(X_test)

            # Evaluate the model
            print(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)

            last_directory = os.path.basename(os.path.dirname(file))

            file_path = f"data/results/random_forest_classification_report_{last_directory}.txt"
            with open(file_path, "w") as f:
                f.write(report)

            print(f"Classification report saved to {file_path}")

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'RF Confusion Matrix for {last_directory}')
            plt.savefig(f'data/results/confusion_matrix_plot_{last_directory}.png')
            plt.close()

def main():
    rf()

if __name__ == "__main__":
    main()