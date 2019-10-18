import torch
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim, nn
import torch.nn.functional as F

import network as nets

class Model:
    def __init__(self, hidden, learning_rate, batch_size):
        self.batch_size = batch_size
        self.net = nets.AutoEncoder(hidden)
        self.net
        #self.opt = optim.Adam(self.net.parameters(), learning_rate)
        self.opt = optim.SGD(self.net.parameters(), learning_rate, momentum=0.9, weight_decay=1e-4)
        self.feature_size = hidden[0] # n_user/n_item

    def run(self, trainset, testlist, num_epoch):
        for epoch in range(1, num_epoch + 1):
            #print "Epoch %d, at %s" % (epoch, datetime.now())
            train_loader = DataLoader(trainset, self.batch_size, shuffle=True, pin_memory=True)
            self.train(train_loader, epoch)
            self.test(trainset, testlist)
    #批训练
    def train(self, train_loader, epoch):
        self.net.train()
        features = Variable(torch.FloatTensor(self.batch_size, self.feature_size))
        masks = Variable(torch.FloatTensor(self.batch_size, self.feature_size))

        for bid, (feature, mask) in enumerate(train_loader):
            if mask.shape[0] == self.batch_size:
                features.data.copy_(feature)
                masks.data.copy_(mask)
            else:
                features = Variable(feature)
                masks = Variable(mask)
            self.opt.zero_grad()
            output = self.net(features)
            loss = F.mse_loss(output* masks, features* masks)
            #loss = F.mse_loss(output, features)
            loss.backward()
            self.opt.step()

        print ("Epoch %d, train end." % epoch)

    def test(self, trainset, testlist):
        self.net.eval()
        x_mat, mask, user_based = trainset.get_mat()
        features = Variable(x_mat)
        xc = self.net(features)
        if not user_based:
            xc = xc.t()
        xc = xc.cpu().data.numpy()

        rmse = 0.0
        for (i, j, r) in testlist:
            rmse += (xc[i][j]-r)*(xc[i][j]-r)
        rmse = math.sqrt(rmse / len(testlist))

        print (" Test RMSE = %f" % rmse)
