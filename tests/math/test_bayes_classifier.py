import unittest
import numpy as np
import pandas as pd

import src.math.bayes_classifier


def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]



class BayesClassifierTestCase(unittest.TestCase):
    def test_bayes_classifier(self):
        np.random.seed(42)
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        data = pd.read_csv('iris.data', names=names)
        train, test = split_train_test(data,0.2)
        nb = src.math.bayes_classifier.NaiveBayes()
        nb.fit(train)
        value = nb.predict(test)
        correct, error = 0, 0
        for num, val in value.items():
            if data['class'][num] == val:
                correct += 1
            else:
                error += 1
        self.assertGreaterEqual(correct/(correct+error), 0.8)


if __name__ == '__main__':
    unittest.main()
