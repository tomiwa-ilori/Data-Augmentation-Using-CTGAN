import pytest
import numpy as np
from bud import read_data
from typing import Callable, Dict, NoReturn


@pytest.fixture
def test_data_input() -> Dict[str, np.dtype]:

    """
    Retrieve the data types for the columns in our pandas dataframe
    """
    data = read_data("Bud-DataScienceTask/bud_ds_data.csv")
    test_data_inputs = data.dtypes.to_dict()
    return test_data_inputs

def test_data_types(test_data_input: Callable) -> NoReturn:

    """
    Create data schema and test that the type equates the data type of our dataset
    """
    schema_type = { 
                    'amount': np.dtype('float64'),
                    'class_': np.dtype('int64'),
                    'date': np.dtype('O'),
                    'description': np.dtype('O'),
                    'id': np.dtype('int64')}
    assert  test_data_input == schema_type 
