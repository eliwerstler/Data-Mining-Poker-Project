# Poker Hand Classification Using Data Mining Techniques

## Overview

This project focuses on performing classification tasks on a **Poker Hand Dataset**, created by Franz Oppacher of Carleton University. The dataset consists of 11 attributes, including 10 predictive attributes representing the rank and suit of five cards and 1 goal attribute representing the class of the poker hand. Our goal was to classify poker hands accurately, ranging from "nothing in hand" to "royal flush." We implemented and evaluated multiple classification algorithms, including **k-nearest neighbors (KNN)**, **decision trees**, and **neural networks**, while optimizing for performance and accuracy through data preprocessing and hyperparameter tuning.

---

## Dataset

- **Attributes**: 10 predictive attributes (rank and suit of 5 cards) and 1 target class (the poker hand).
- **Classes**: 0 ("nothing in hand"), 1 ("one pair"), up to 9 ("royal flush").
- **Instances**: A combined set of 1,025,010 instances.

We explored the dataset’s class distribution, and since the dataset was highly imbalanced (most hands were "nothing in hand" or "one pair"), we utilized **stratified sampling** during our data splitting to ensure balanced class representation in training and testing sets.

---

## Data Preprocessing

Key steps taken in preprocessing:
- **Outliers and Missing Data**: The dataset had no missing data or outliers, as expected.
- **Dimensionality Reduction**: We treated different permutations of a hand as the same hand by sorting the cards, which drastically reduced the dimensionality of the dataset and improved classification performance.
- **Normalization**: While normalization is typically useful, it did not enhance model performance due to the ordinal nature of the rank and suit attributes, so we opted against it.

---

## Algorithms Used

1. **K-Nearest Neighbors (KNN)**:
   - Initially slow with low accuracy (~60%).
   - After sorting the cards to reduce dimensionality, performance improved to an **accuracy of over 80%**.
   - Best performance observed with `k = 4` and `k = 6`.

2. **Decision Trees**:
   - Initial performance was similar to KNN (~60%).
   - After sorting, decision trees reached an accuracy of **~70%** with lower runtime.

3. **Neural Networks**:
   - Neural networks showed **high accuracy (~99%)** but had the trade-off of **significantly longer runtimes**.
   - Performance degraded with smaller train splits, but this method still yielded the best accuracy results overall.

---

## Results

Through our experimentation with multiple algorithms:
- Sorting the cards improved both KNN and decision tree runtimes and accuracy.
- **K-Nearest Neighbors** emerged as a well-balanced option between accuracy and runtime after sorting the cards, reaching an accuracy of **~82%**.
- **Neural Networks** provided the highest accuracy but were computationally expensive, taking significantly longer to run.
  
By implementing dimensionality reduction techniques (sorting the cards), we achieved more efficient models, with **KNN** proving to be an effective and fast solution for this dataset after adjustments.

---

## Tools & Libraries Used

- **Python**: For general implementation.
- **NumPy**, **Pandas**: For data manipulation and preprocessing.
- **Matplotlib**: For data visualization.
- **Scikit-learn**: For implementing KNN, decision trees, and neural networks.

---

## Collaboration

- Eli Werstler and William Schimitsch worked collaboratively on data analysis and classification tasks.
- Eli led the **neural network classification** and **numerical analysis**, while William handled **data sorting** and helped improve the performance of KNN and decision tree algorithms.

---

## Conclusion

This project demonstrated the importance of data preprocessing, particularly dimensionality reduction, in improving classification models for large datasets. **K-nearest neighbors** proved to be an effective and efficient method for classifying poker hands after preprocessing. Though **neural networks** yielded the highest accuracy, they were not optimal for time-sensitive tasks due to their computational demands.

---

## References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)
- [Matplotlib Documentation](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html)
