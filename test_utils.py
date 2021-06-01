import unittest
from utils import f1, read_text_data, read_data_span

class TestUtilMethods(unittest.TestCase):

    def test_f1(self):
        self.assertEqual(0.7142857142857143, f1([1,2,3,4,5,6,7,8,9],[2,3,4,5,6]))

    def test_read_text_data(self):
        test_string = 'What a knucklehead. How can anyone not know this would be offensive??'
        self.assertEqual(test_string, read_text_data('data/tsd_train_readable.csv')[3])

    def test_read_data_span(self):
        test_span = '[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]'
        self.assertEqual(test_span, read_data_span('data/tsd_train_readable.csv')[3])

if __name__ == '__main__':
    unittest.main()