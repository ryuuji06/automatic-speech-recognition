
"""
Defines a class that manages the data, constituting a bridge between
obtaining the input data (features) and target labels and providing
the data to the network batch-wise.
"""

import json
import random
import numpy as np


class DatasetManager():

    def __init__(self, train_file=None, valid_file=None, test_file=None,
        minibatch_size=20, feat_size=None, feature_func=None, label_func=None ):
        """Params:"""

        # data set parameters
        self.feat_size = feat_size
        self.feats_mean = np.zeros((self.feat_size,))
        self.feats_std = np.ones((self.feat_size,))
        self.minibatch_size = minibatch_size
        
        # feature function must have one input (filepath) and one output (feature array)
        # always normalize, but with standard values, it makes no difference
        assert feature_func is not None, "No feature function was passed."
        self.featurize = lambda path: self.normalize(feature_func(path))

        # function to convert label to numeric value
        if label_func is None:
            self.labelize = lambda x: x
        else:
            self.labelize = label_func

        # load files
        if train_file is not None:
            self.load_metadata_from_desc_file(train_file,'train')
        if valid_file is not None:
            self.load_metadata_from_desc_file(valid_file,'validation')
        if test_file is not None:
            self.load_metadata_from_desc_file(test_file,'test')

        # current index for iterative training
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0


    # the JSON key names are specific to this audio problem
    def load_metadata_from_desc_file(self, filepath, partition):
        """
        Read metadata from a JSON-line file (possibly takes long, depending on the filesize)
        Params:
            filepath (str):  Path to a JSON-line file that contains labels and paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
        """
        input_paths, labels = [], []
        with open(filepath) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    input_paths.append(spec['key'])
                    labels.append(spec['text'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or (KeyError,json.decoder.JSONDecodeError),
                    # depending on json module version
                    print('Error reading line #{}: {}' .format(line_num, json_line))
        if partition == 'train':
            self.train_input_paths = input_paths
            self.train_labels = labels
        elif partition == 'validation':
            self.valid_input_paths = input_paths
            self.valid_labels = labels
        elif partition == 'test':
            self.test_input_paths = input_paths
            self.test_labels = labels
        else:
            raise Exception("Invalid partition to load metadata. "
             "Must be train/validation/test")

    def load_data(self, filepath, partition):
        self.load_metadata_from_desc_file(filepath, partition)

    def get_sample(self, sample_num, partition):
        if partition == 'train':
            assert sample_num < len(self.train_input_paths), 'Sample out of range!'
            return self.featurize(self.train_input_paths[sample_num]), self.labelize(self.train_labels[sample_num])
        elif partition == 'validation':
            assert sample_num < len(self.valid_input_paths), 'Sample out of range!'
            return self.featurize(self.valid_input_paths[sample_num]), self.labelize(self.valid_labels[sample_num])
        elif partition == 'test':
            assert sample_num < len(self.test_input_paths), 'Sample out of range!'
            return self.featurize(self.test_input_paths[sample_num]), self.labelize(self.test_labels[sample_num])
        else:
            raise Exception("Invalid partition to load metadata. "
             "Must be train/validation/test")

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    # stacking features along one dimension is specific to this problem
    def set_feature_normalization(self, precomputed=None, k_samples=200):
        """ Estimate the mean and std of training set for feature normalization
        and change feature function to include normalization
        Params:
            precomputed: tuple with feats_mean and feats_std
            k_samples (int): Use this number of samples for estimation
        """
        if precomputed==None:
            print('Feature normalization: computing mean and std')
            k_samples = min(k_samples, len(self.train_input_paths))
            samples = random.sample(self.train_input_paths, k_samples)
            feats = []
            for i,s in enumerate(samples):
                feats.append(self.featurize(s))
                if (i+1)%20==0:
                    print(' sample %d'%(i+1))
            feats = np.vstack(feats) # this is specific to audio features
            self.feats_mean = np.mean(feats, axis=0)
            self.feats_std = np.std(feats, axis=0)
        else:
            self.feats_mean = precomputed[0]
            self.feats_std = precomputed[1]

    # get_batch, used within 'next_train()', 'next_valid()' and 'next_test()'
    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data"""

        if partition == 'train':
            input_paths = self.train_input_paths
            cur_index = self.cur_train_index
            labels = self.train_labels
        elif partition == 'valid':
            input_paths = self.valid_input_paths
            cur_index = self.cur_valid_index
            labels = self.valid_labels
        elif partition == 'test':
            input_paths = self.test_input_paths
            cur_index = self.cur_test_index
            labels = self.test_labels
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

        # compute and store features of the next N=`minibatch_size` samples
        features = [ self.featurize(a) for a in 
            input_paths[cur_index:cur_index+self.minibatch_size]]

        # calculate necessary sizes
        max_length = max([ features[i].shape[0] 
            for i in range(0, self.minibatch_size) ])
        max_string_length = max([ len(labels[cur_index+i]) 
            for i in range(0, self.minibatch_size) ])
        
        # initialize the minibatch rectangular arrays (features and numeric labels)
        # (input and label lengths are also input to the RNN)
        X_data = np.zeros([ self.minibatch_size, max_length, self.feat_size ])
        num_labels = np.ones([self.minibatch_size, max_string_length]) * 28 # maps everything to `28`
        input_length = np.zeros([self.minibatch_size, 1])
        label_length = np.zeros([self.minibatch_size, 1])
        
        # insert elements on the minibatch array
        for i in range(0, self.minibatch_size):
            # calculate X_data & input_length
            feat = features[i]
            input_length[i] = feat.shape[0]
            X_data[i, :feat.shape[0], :] = feat

            # calculate labels & label_length
            num_label = self.labelize(labels[cur_index+i]) 
            num_labels[i, :len(num_label)] = num_label
            label_length[i] = len(num_label)
 
        # return the arrays
        outputs = {'ctc': np.zeros([self.minibatch_size])}
        inputs = {'input_seq': X_data, 
                  'label_seq': num_labels, 
                  'input_length': input_length, 
                  'label_length': label_length 
                 }
        return (inputs, outputs)


    # get next batches, shuffle if end is reached
    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data"""
        if partition == 'train':
            self.train_input_paths, self.train_labels = shuffle_data(self.train_input_paths, self.train_labels)
        elif partition == 'valid':
            self.valid_input_paths, self.valid_labels = shuffle_data(self.valid_input_paths, self.valid_labels)
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

    def next_train(self):
        """ Obtain a batch of training data"""
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= len(self.train_labels) - self.minibatch_size:
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret    

    def next_valid(self):
        """ Obtain a batch of validation data"""
        while True:
            ret = self.get_batch('valid')
            self.cur_valid_index += self.minibatch_size
            if self.cur_valid_index >= len(self.valid_labels) - self.minibatch_size:
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
            yield ret

    def next_test(self):
        """ Obtain a batch of test data"""
        while True:
            ret = self.get_batch('test')
            self.cur_test_index += self.minibatch_size
            if self.cur_test_index >= len(self.test_labels) - self.minibatch_size:
                self.cur_test_index = 0
            yield ret



# auxiliar function to shuffle data
def shuffle_data(input_paths, labels):
    """ Shuffle the data (called after making a complete pass through 
        training or validation data during the training process)
    """
    p = np.random.permutation(len(input_paths))
    input_paths = [input_paths[i] for i in p] 
    labels = [labels[i] for i in p]
    return input_paths, labels



