import unittest
from house_prices.preprocess import divide

"""
unite test
integrate test
1-to-1 test
            __
           /  \
          /    \
         /      \
        /        \
       /          \
      /___________ \
     /  1-to-1 test \
    /________________\
   / Integration test \
  /____________________\
 /     unite teste      \
/------------------------\
"""

class MyTestCase(unittest.TestCase):
    # def test_something(self):
    #     self.assertEqual(True, False)  # add assertion here

    """
    * No code if no red test
    * Only code that resolve the error message
    """

    """
    Test list for divide
    1. normal behavior, positive values
        input : a = 15, b =3
        output: 5
    
    2. normal behavior, negative values input and negative output
        input: a = 15, b = -3
        output: -5
        
    3. normal bahavior negative input positive output
        input: a = -15, b = -3
        output: 5
    
    4. input non numerical value type
        input: a = 15, b = '-3'
        output: TypeError exception
    
    5. divide by zero
        input: a = 15, b = 0
        output:ZeroDivisionError except 
    """
    def test_divide_normal_behavior_with_positive_values__(self):

        # given
        a = 15
        b = 3
        expected = 5

        # when
        result = divide(a, b)

        # then
        self.assertEqual(expected, result)

if __name__ == '__main__':
    unittest.main()
