# Classify YouTube Category Based on Video Tags

## Introduction
This project aims to automate the categorization of YouTube videos into music or sports categories, based on video tags. This process can enhance the efficiency of categorization and aid in targeted advertising.

## Data Source
We utilized a dataset from Kaggle, focusing on video records from the United States. The dataset includes around 30,000 records for each category. Non-English characters were removed, and the data was tokenized and stored in separate text files.

## Methods
### Naive Bayes Model
- **Algorithm**: We employed the Naive Bayes model, using raw count and TF-IDF for word embeddings.
- **Assumptions**: Assumes independence of features and sequence irrelevance.
- **Performance Metrics**: Accuracy and time efficiency.

### Recurrent Neural Network (RNN)
- **Architecture**: Embedding layer followed by an RNN layer and a linear layer.
- **Hyperparameters**: Hidden size, batch size, learning rate, etc.
- **Training Process**: Cross-entropy loss with Adam optimizer over 20 epochs.
- **Performance Metrics**: Loss, accuracy, and training time.

## Results
### Real Data
- **Naive Bayes Model**: Achieved high accuracy (~96%) with both raw count and TF-IDF embedding methods. TF-IDF was more time-efficient.
- **RNN Model**: Reached an accuracy of 89.57% after 20 epochs. Demonstrated effective learning patterns and time consistency.

### Synthetic Data
- **Naive Bayes Model**: Slightly higher accuracy (~99%) than with real data. Time efficiency improved.
- **RNN Model**: Showed moderate success (61.55% accuracy) but faced challenges in learning patterns and convergence.

## Conclusion
- **Naive Bayes Model**: Exhibits high accuracy and time efficiency. Potential improvements include ensemble methods and smoothing for zero probabilities.
- **RNN Model**: Effective in capturing sequential dependencies but faces limitations with long-term dependencies and variable-length sequences. Performance heavily relies on data quality.