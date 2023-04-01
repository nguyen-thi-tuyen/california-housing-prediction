import unittest
import os

from src.data.make_dataset import generate_train_test_addresses


class TestGenerateTrainTestAddresses(unittest.TestCase):
    def test_output(self):
        """
        Test that generate_train_test_addresses returns a dictionary with the output paths
        """
        data = 'mypath'
        result = generate_train_test_addresses(data)
        expected_result = {'train_feature_path': 'mypath/X_train.csv',
                           'test_feature_path': 'mypath/X_test.csv',
                           'train_target_path': 'mypath/y_train.csv',
                           'test_target_path': 'mypath/y_test.csv'}
        self.assertEqual(result, expected_result)
        os.rmdir(data)

    def test_folder_creation(self):
        """
        Test that generate_train_test_addresse creates the parent directory when
        it does not exist
        """
        data = 'mypath'
        result = generate_train_test_addresses(data)
        self.assertEqual(os.path.exists(data), True)
        os.rmdir(data)

if __name__ == '__main__':
    unittest.main()