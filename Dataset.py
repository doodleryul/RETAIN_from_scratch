###########################
### [0] Common function ###
###########################

def dumpingFiles(basePath, outFilename, files):
    import os, pickle
    dumpingPath = os.path.join(basePath, outFilename)
    print("Dumping at..", dumpingPath)
    with open(dumpingPath, 'wb') as outp:
        pickle.dump(files, outp, -1)
        
def loadingFiles(basePath, filename):
    import os, pickle
    loadingPath = os.path.join(basePath, filename)
    print("Loading at..", loadingPath)
    with open(loadingPath, 'rb') as f:
        p = pickle.load(f)
    return p
    
##############################
### [1] Get dataset format ###
##############################

def cut_label(treat_add_label):
    treat = []
    for seqs in treat_add_label:
        treat.append(seqs[:-1])
    return treat

def truncated_data(dx_seqs, cut_num):
    truncated_d_seq = []
    for d_seq in dx_seqs:
        if cut_num<=len(d_seq):
            truncated_d_seq.append(d_seq)
        else:
            print(d_seq)
    return truncated_d_seq

def reverse_seq(seqs):
    import numpy as np
    reversed_seqs = []
    for seq in seqs:
        reversed_seqs.append(np.array(seq)[::-1].tolist())
    return reversed_seqs

def code_to_id(data):
    from itertools import chain
    visit_length = []
    for total_visit_seq in data:
        visit_length.append(len(total_visit_seq))
    max_visit_length = max(set(visit_length))
    code_to_id = {code: i for i, code in enumerate(set(chain.from_iterable(chain.from_iterable(data))))}
    print('code_size: ', len(code_to_id))
    return code_to_id, max_visit_length

class Generate_batch():
    def __init__(self, inputs, labels, visit_times):
        import numpy as np
        self._num_examples = len(inputs)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._inputs = inputs
        self._labels = labels
        self._visit_times = visit_times
        
    def _decompress_sparseM(self, sparse_list):
        decompressed_inputs = []
        for pid_list in sparse_list:
            inputs = pid_list.toarray()
            decompressed_inputs.append(inputs)
        return decompressed_inputs
            
    def _shuffle(self, list1, list2, list3):
        import sklearn as sk
        return sk.utils.shuffle(list1, list2, list3)

    def next_batch(self, batch_size, shuffle=True):
        import numpy as np
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        if end <= self._num_examples:
            decompressed_inputs = self._decompress_sparseM(self._inputs[start:end])
            batch_inputs, batch_labels, batch_visit_times = self._shuffle(decompressed_inputs, self._labels[start:end], self._visit_times[start:end])
            return np.array(batch_inputs), np.array(batch_labels), np.array(batch_visit_times)
        else:
            self._epochs_completed += 1
            num_of_short = batch_size-(self._num_examples-start)
            num_of_extra_batch = num_of_short // self._num_examples
            num_of_extra_example = num_of_short % self._num_examples
            self._epochs_completed += num_of_extra_batch
            self._index_in_epoch = num_of_extra_example
            
            tmp_inputs = self._inputs[start:]; tmp_labels = self._labels[start:]; tmp_visit_times = self._visit_times[start:]    
            if shuffle:
                self._inputs, self._labels, self._visit_times = self._shuffle(self._inputs, self._labels, self._visit_times)
            batch_inputs = tmp_inputs + self._inputs*num_of_extra_batch + self._inputs[0:num_of_extra_example]
            batch_labels = tmp_labels + self._labels*num_of_extra_batch + self._labels[0:num_of_extra_example]
            batch_visit_times = tmp_visit_times + self._visit_times*num_of_extra_batch + self._visit_times[0:num_of_extra_example]
            return np.array(self._decompress_sparseM(batch_inputs)), np.array(batch_labels), np.array(batch_visit_times)
        
class concat_Dataset():
    def __init__(self, t_dataset, c_dataset):
        self._t_dataset = t_dataset
        self._c_dataset = c_dataset
        
    def next_batch(self, batch_size, c_ratio=1):
        import sklearn as sk
        import numpy as np
        
        t_batch_size = int(batch_size/(1+c_ratio))
        c_batch_size = batch_size-t_batch_size
        
        t_inputs, t_labels, t_visit_times = self._t_dataset.next_batch(t_batch_size)
        c_inputs, c_labels, c_visit_times = self._c_dataset.next_batch(c_batch_size)
        batch_inputs = np.concatenate([t_inputs, c_inputs], axis=0)
        batch_labels = np.concatenate([t_labels, c_labels], axis=0)
        batch_visit_times = np.concatenate([t_visit_times, c_visit_times], axis=0)
        
        return sk.utils.shuffle(batch_inputs, batch_labels, batch_visit_times)

def _convert_to_sparseMultiHot(data, code_to_id, max_visit, labeling):
    from scipy.sparse import csr_matrix
    import numpy as np

    pid_multihot = []
    labels = []
    for pid, dxs in enumerate(data):
        pid_row = []
        pid_col = []
        for i, codes in enumerate(dxs):
            for c in codes:
                pid_row.append(i)
                pid_col.append(code_to_id[c])
        visit_times=len(dxs)
        if labeling == 'T': label = [[1,0]]*max_visit
        elif labeling == 'C': label = [[0,1]]*max_visit
        pid_multihot.append([csr_matrix((np.ones(len(pid_col), np.float32), (pid_row, pid_col)), 
                                       shape=(max_visit, len(code_to_id))), label, visit_times])
    return np.array(pid_multihot)[:,0].tolist(), np.array(pid_multihot)[:,1].tolist(), np.array(pid_multihot)[:,2].tolist()   

def _split(inputs, labels, visit_times):
    import sklearn as sk
    
    split_idx = [int(len(inputs)*0.75), int(len(inputs)*0.85)]
    inputs, labels, visit_times = sk.utils.shuffle(inputs, labels, visit_times)
    inputs_list = [inputs[:split_idx[0]], inputs[split_idx[0]:split_idx[1]], inputs[split_idx[1]:]]
    labels_list = [labels[:split_idx[0]], labels[split_idx[0]:split_idx[1]], labels[split_idx[1]:]]
    visit_times_list = [visit_times[:split_idx[0]], visit_times[split_idx[0]:split_idx[1]], visit_times[split_idx[1]:]]
    return inputs_list, labels_list, visit_times_list
    
#################################
### [2] RETAIN dataset format ###
#################################

def RETAIN_datasets(t_dataset, c_dataset, code_to_id, max_visit):
    t_multihot = _convert_to_sparseMultiHot(t_dataset, code_to_id, max_visit, labeling='T')
    c_multihot = _convert_to_sparseMultiHot(c_dataset, code_to_id, max_visit, labeling='C')
    t_inputs_list, t_labels_list, t_visit_times_list = _split(t_multihot[0], t_multihot[1], t_multihot[2])
    c_inputs_list, c_labels_list, c_visit_times_list = _split(c_multihot[0], c_multihot[1], c_multihot[2])

    class _generate_dataset(): pass
    datasets = _generate_dataset()
    datasets.train = concat_Dataset(Generate_batch(t_inputs_list[0], t_labels_list[0], t_visit_times_list[0]),
                                    Generate_batch(c_inputs_list[0], c_labels_list[0], c_visit_times_list[0]))
    datasets.validation = concat_Dataset(Generate_batch(t_inputs_list[1], t_labels_list[1], t_visit_times_list[1]),
                                         Generate_batch(c_inputs_list[1], c_labels_list[1], c_visit_times_list[1]))
    datasets.test = concat_Dataset(Generate_batch(t_inputs_list[2], t_labels_list[2], t_visit_times_list[2]),
                                   Generate_batch(c_inputs_list[2], c_labels_list[2], c_visit_times_list[2]))
    return datasets