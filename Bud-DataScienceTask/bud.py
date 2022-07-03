
# import required packages
import random
import pandas as pd
import datetime as dt
from sdv.tabular import CTGAN
from strictly_typed_pandas import DataSet



class Schema:
    """
        Validate data input using python typing
    """
    date: dt
    amount: float
    description: str
    class_: int
    id: int
    
def read_data(data: DataSet[Schema]) -> DataSet[Schema]:

    """
        Read data into pandas
    """
    data = pd.read_csv(data)
    return data



def dataAugmentation(num_samples):

    """ 
        The augmentation model used in this solution is the CTGAN model which was presented in the paper below.
        Paper for the implementation can be found here: https://arxiv.org/pdf/1907.00503.pdf
        This model has been trained and we only load the model in this function.
        Uncomment line 35, 38, and 41 to retrain the model and save.
    """

    # instantiate the model using the CTGAN method
    # giving that 'id' is a unique field, we decided to inform the model that we want the id field to be unique
    #model = CTGAN(primary_key='id') 

    # fit the model on the original dataset
    #model.fit(data)

    # save model after training; for latency 
    #model.save('Bud-DataScienceTask/models/ctganModel.pkl') 

    # load the saved moel and use it to create synthetic versions of the original dataset
    load_model = CTGAN.load('Bud-DataScienceTask/models/ctganModel.pkl')
    new_data = load_model.sample(num_samples)

    return new_data   


def randomNumber():
    """
        Generate random number
    """
    random_num = round(random.uniform(0, 1), 1)
    return random_num


def transactionalDataGenerator(p=1.0, batch_size=3):
    """
        Create augmentated data generator
    """
    random_number = randomNumber()
    if random_number >= p:
        print ("----- Generating Augmentated Data -----")
        yield dataAugmentation(batch_size)
    else:
        print ("----- Printing Random Sample of Original Data -----")
        yield data.sample(batch_size)

if __name__ == "__main__":

    # call method to read data
    data = read_data('Bud-DataScienceTask/datasets/bud_ds_data.csv')

    # call method to generate new data according to batch sixe
    gen = transactionalDataGenerator()

    print (next(gen))

    

    