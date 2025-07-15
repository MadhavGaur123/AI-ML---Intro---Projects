🧠 AI/ML + Computer Vision Projects Portfolio

This repository showcases a collection of applied AI/ML and deep learning projects, ranging from classification and regression to object detection using YOLOv8. It demonstrates both breadth and depth in ML/AI with real-world datasets and tasks.
📁 Contents
Category	Project Name	Description
Machine Learning	Decision Tree Disease Detection	Heart disease classification
Machine Learning	Decision Tree Loan Approval	Loan prediction with ROC curve
Machine Learning	K-Means Customer Segmentation	Customer grouping based on spending
Machine Learning	Linear Regression Housing Prices	Regression with 3D visualization
Machine Learning	Sentiment Analysis using TF-IDF	NLP-based review classifier
Deep Learning	YOLOv8 Helmet Detection System	Object detection (helmet/no-helmet)
🤖 AI/ML Mini Projects Collection

This section contains five concise and practical machine learning projects developed using Python, Scikit-learn, Pandas, NumPy, and Matplotlib.
🌐 Project List Overview
Project Name	Technique	Focus
DecisionTreeDiseaseDetection	Classification	Heart disease prediction
DecisionTreeLoanApproval	Classification + ROC	Loan approval model with ROC
K-means-clustering	Unsupervised Clustering	Customer segmentation
LinearRegressionModel	Regression	Housing price prediction (3D plot)
SentimentAnalysis	NLP + Classification	Text sentiment analysis (TF-IDF)
✅ Decision Tree: Disease Detection

    Predicts heart disease using features like cholesterol and heart rate.

    Implements Decision Tree and Random Forest classifiers.

    Evaluates with confusion matrix and ROC curve.

✅ Decision Tree: Loan Approval with ROC

    Classifies loan approvals based on structured inputs.

    Calculates TPR and FPR manually.

    Plots a pseudo ROC curve.

✅ K-Means Clustering: Customer Segmentation

    Groups mall customers using annual income and spending score.

    Visualizes colored clusters and centroids.

    Uses Elbow Method for K optimization.

✅ Linear Regression: Housing Price Prediction

    Predicts price from square footage and number of bedrooms.

    3D surface plot with real-time regression visualization.

    Metrics: R² Score, Mean Squared Error.

✅ Sentiment Analysis using TF-IDF

    Classifies text reviews as positive or negative.

    Uses TF-IDF vectorization + Logistic Regression.

    Includes NLTK preprocessing (lemmatization, stopwords).

🔧 Dependencies

Install ML dependencies:

pip install pandas numpy scikit-learn matplotlib nltk

🪖 YOLOv8 Helmet Detection System

This project implements a full deep learning pipeline for detecting helmet usage using YOLOv8. It includes:

    Data integrity verification

    Model training

    Image/video inference

    Model export

📁 File Structure

├── check.py                # Dataset sanity check (images ↔ labels)
├── yolov8.py               # Trains YOLOv8 on custom dataset
├── SelfTrainedmodel.py     # Inference on a single image
├── testvideo.py            # Inference on video stream (e.g. CCTV)
├── data.yaml               # YOLO training configuration file
├── YOLO_helmet_training/   # Output training logs and weights
├── trained_helmet_model.pt # Exported TorchScript model

🔍 1. Dataset Verification — check.py

Ensures all image files have corresponding label .txt files (YOLO format) and vice versa.

python check.py

Output:

    ✅ All matched

    🚨 Reports missing labels or images

🧠 2. Model Training — yolov8.py

Trains a YOLOv8 model with the ultralytics library.
Key Parameters:

    Epochs: 50

    Batch size: 16

    Image size: 640

    Base model: yolov8n.pt

    Project: "YOLO_helmet_training"

python yolov8.py

Output:

    Training logs in /YOLO_helmet_training/

    Best weights exported to TorchScript (trained_helmet_model.pt)

🖼️ 3. Inference on Image — SelfTrainedmodel.py

Loads a .jpg image and runs inference on the trained helmet detection model.

python SelfTrainedmodel.py

    Displays bounding boxes and class labels

    Runs on GPU if available (torch.cuda)

🎥 4. Inference on Video — testvideo.py

Runs real-time detection on video footage and saves annotated output.

python testvideo.py

Features:

    Helmet detection with bounding box and label

    Warning overlay if no helmet is detected

    Output saved as output_helmet_detection.mp4

🧠 Dependencies

Install YOLOv8 and computer vision libraries:

pip install ultralytics torch opencv-python

🏁 Final Thoughts

These projects combine both machine learning fundamentals and modern computer vision techniques into a single repository. They serve as an ideal base for:

    Academic demonstration

    Technical portfolios

    Real-world applications (e.g., safety monitoring, user profiling)
