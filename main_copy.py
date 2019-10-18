import sys
sys.path.append('../../')
import setdata as sd  
import model
from datetime import datetime
from data_loader import load_data
#from torch.utils import data

# parameters
rank = 100
batch_size = 128
user_based = False

if __name__ == '__main__':
    start = datetime.now()
    train_list, test_list, n_user, n_item = load_data('ratings', 0.9)
    trainset = sd.Dataset(train_list, n_user, n_item, user_based)
    if user_based :
        h = n_item
    else:
        h = n_user

    mod = model.Model(hidden=[h, rank*3],
                      learning_rate = 0.2,
                      batch_size=batch_size)

    mod.run(trainset, test_list, num_epoch=500)

    end = datetime.now()
    print ("Total time: %s" % str(end-start))
