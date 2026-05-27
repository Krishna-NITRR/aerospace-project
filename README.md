# aerospace-project
I am Krishna Mahawar, writing readme file!

I want to make a EDA + MATPLOTLIB + Scikit-learn + Machine learning model (4) Project that is  using NASA's Near-Earth Objects (NEO) dataset from Kaggle, classified as hazardous vs. non-hazardous using scikit-learn, with rich Matplotlib visualizations.

Dataset
NASA Nearest Earth Objects - from NASA's NeoWs API via Kaggle

~90,000 asteroids, features: estimated diameter, relative velocity, miss distance, absolute magnitude, hazardous label
Kaggle link: kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects

Thanks to kaggle and Google! 

What This Project Does

- Downloads the NASA asteroid dataset automatically
- Visualizes class distribution, diameter spread, and velocity patterns
- Trains 4 machine learning classifiers
- Compares them using confusion matrices and ROC curves
- Shows feature importance and PCA visualization

- View the Results

ChartWhat It ShowsChart 
1Class distribution, diameter histogram, miss distance vs velocityChart 
2Confusion matrices for all 4 classifiersChart 
3ROC curves comparing all classifiersChart 
4Feature importance from Random ForestChart 
5PCA 2D visualization of predictions

- Models Used

| Model               | Accuracy | AUC    |
|---------------------|----------|--------|
| Logistic Regression | 90.33%   | 0.8812 |
| K-Nearest Neighbors | 90.21%   | 0.8731 |
| Random Forest       | 91.67%   | 0.9330 |
| Gradient Boosting   | 91.50%   | 0.9210 |
