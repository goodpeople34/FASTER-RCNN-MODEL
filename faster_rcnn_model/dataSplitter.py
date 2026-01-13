import random
from config import DATASET_ID, TRAIN_SIZE, TRAIN_FILE_PATH, TEST_FILE_PATH

class DatasetSplitter:
    def __init__(self, dataset_id, train_size, train_file, test_file):
        self.dataset_id = dataset_id
        self.train_size = train_size
        self.train_file = train_file
        self.test_file = test_file

        self.train_ids = []
        self.test_ids = []

    def generate_ids(self):
        self.train_ids = random.sample(range(1, self.dataset_id), self.train_size)
        self.test_ids = list(set(range(1, self.dataset_id)) - set(self.train_ids))

        self.train_ids.sort()
        self.test_ids.sort()

    def save_to_file(self):
        with open(self.train_file, 'w') as f:
            f.write('\n'.join(map(str,self.train_ids)))


        with open(self.test_file, 'w') as f:
            f.write('\n'.join(map(str,self.test_ids)))

    def run(self):
        self.generate_ids()
        self.save_to_file()


data_split = DatasetSplitter(DATASET_ID, TRAIN_SIZE, TRAIN_FILE_PATH, TEST_FILE_PATH)

data_split.run()