import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
import pdb
from scipy.optimize import linear_sum_assignment
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# '/home/lz/Workspace/ZSL/data/Animals_with_Attributes2',

parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b","--batch_size",type = int, default = 64)
parser.add_argument("-e","--episode",type = int, default= 5000000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default=1e-5)#default = 1e-5)
parser.add_argument("-g","--gpu",type=int, default=1)
args = parser.parse_args()


# Hyper Parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

def get_nn_sequentials(n1, n2, out_size):
    seq = nn.Sequential(
        nn.Linear(n1 * 2048, n2*out_size),
        # nn.Linear(n1 * 1000, n2*out_size),
        # nn.Linear(n1 * 512, n2 * out_size),
        nn.BatchNorm1d(n2*out_size),
        nn.ReLU()
    )
    return seq
class CoarseNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,n1, n2, out_size):
        super(CoarseNetwork, self).__init__()
        # self.lin_fc123 = get_nn_sequentials(n1, n2, out_size)
        self.fc1 = nn.Linear(n1, n2)
        self.fc2 = nn.Linear(n2, out_size)

    def forward(self,x):

        # x1 = self.lin_fc123(x)
        x1 = F.relu(self.fc1(x))
        x2 = F.softmax(self.fc2(x1))

        return x1, x2

class FineNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,out_size):
        super(FineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,out_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.sigmoid(self.fc2(x))

        return x

class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc2(x))

        return x
# see_f:64*2048, seen_att_f:40*2048
def seendist(seen_f,seen_att_f,seen_l):
    att_f = seen_att_f[seen_l, :]
    # att_f = seen_att_f[:, :,seen_l]

    L2_loss=((seen_f - att_f) ** 2).sum() / ((40) * 2) ## L2_Loss of seen classes


    return L2_loss
# seen_unseen_f:64*2048, all_att_f:50*2048
def seenAndunseendist(seen_unseen_f, all_att_f, test_seen_id, test_unseen_id):


    DIS=torch.zeros((32,40)).cuda()
    for A_id,x in enumerate(seen_unseen_f[:32,:]):
        for B_id,y in enumerate(all_att_f[test_seen_id,:]):
            dis=((x-y)**2).sum()
            DIS[A_id,B_id]=dis
    matching_loss=0
    cost=DIS.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    for i,x in enumerate(row_ind):
        matching_loss+=DIS[row_ind[i],col_ind[i]]

    DIS2 = torch.zeros((32, 10)).cuda()
    for A_id, x in enumerate(seen_unseen_f[32:64,:]):
        for B_id, y in enumerate(all_att_f[test_unseen_id,:]):
            dis = ((x - y) ** 2).sum()
            DIS2[A_id, B_id] = dis
    cost = DIS2.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    for i, x in enumerate(row_ind):
        matching_loss += DIS2[row_ind[i], col_ind[i]]

    tot_loss=matching_loss*0.0003

    return tot_loss
def main():
    # step 1: init dataset
    print("init dataset")
    
    dataroot = './data'
    dataset = 'AwA2_data'
    image_embedding = 'res101' 
    class_embedding = 'att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    # image_files = matcontent['image_files'].squeeze()
    # matcontent = sio.loadmat('data/my_data/AlexNet_features_labels.mat')
    # label = matcontent['labels'].astype(int).squeeze()
    # feature = matcontent['features'].squeeze()
    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    # trainval_loc, test_seen_loc, test_unseen_loc = [],[],[]
    #
    # unseen_id = np.array([25,39,15,6,42,14,18,48,34,24])-1
    # for ind, item in enumerate(label):
    #     if item in unseen_id:
    #         test_unseen_loc.append(ind)
    #     elif ind % 5 == 0:
    #         test_seen_loc.append(ind)
    #     else:
    #         trainval_loc.append(ind)
  
    attribute = matcontent['original_att'].T
    all_attributes = np.array(attribute)

    train_seen_feature = feature[trainval_loc] # train_features
    train_seen_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_seen_label] # train attributes
    
    test_unseen_feature = feature[test_unseen_loc]  # test_feature
    test_unseen_label = label[test_unseen_loc].astype(int) # test_label

    test_seen_feature = feature[test_seen_loc]  #test_seen_feature
    test_seen_label = label[test_seen_loc].astype(int) # test_seen_label

    test_unseen_id = np.unique(test_unseen_label)   # test_id
    att_unseen_pro = attribute[test_unseen_id]      # test_attribute
    test_seen_id = np.unique(test_seen_label)
    att_seen_pro = attribute[test_seen_id]

    # visual error info

    #train seen
    labels = torch.from_numpy(train_seen_label)
    features = torch.from_numpy(train_seen_feature)
    train_seen_data = TensorDataset(features, labels)
    train_seen_loader = DataLoader(train_seen_data, batch_size=BATCH_SIZE, shuffle=True)

    #test seen
    labels = torch.from_numpy(test_seen_label)
    features = torch.from_numpy(test_seen_feature)
    test_seen_data = TensorDataset(features, labels)
    test_seen_loader = DataLoader(test_seen_data, batch_size=32, shuffle=True)

    # test unseen
    labels = torch.from_numpy(test_unseen_label)
    features = torch.from_numpy(test_unseen_feature)
    test_unseen_data = TensorDataset(features, labels)
    test_unseen_loader = DataLoader(test_unseen_data, batch_size=32, shuffle=True)

    # init network
    print("init networks")
    coarse_network = CoarseNetwork(2048,2048,2)
    attribute_network = AttributeNetwork(85,1024,2048)
    fine_network = FineNetwork(2048+2048,512,1)

    coarse_network.cuda()
    attribute_network.cuda()
    fine_network.cuda()

    coarse_network_optim = torch.optim.Adam(coarse_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    coarse_network_scheduler = StepLR(coarse_network_optim, step_size=200000, gamma=0.5)
    attribute_network_optim = torch.optim.Adam(attribute_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    attribute_network_scheduler = StepLR(attribute_network_optim, step_size=200000, gamma=0.5)
    fine_network_optim = torch.optim.Adam(fine_network.parameters(),lr=LEARNING_RATE)
    fine_network_scheduler = StepLR(fine_network_optim,step_size=200000,gamma=0.5)

    # if os.path.exists("./models/attribute_network.pkl"):
    #     attribute_network.load_state_dict(torch.load("./models/attribute_network.pkl"))
    #     print("load attribute network success")
    # if os.path.exists("./models/fine_network.pkl"):
    #     fine_network.load_state_dict(torch.load("./models/fine_network.pkl"))
    #     print("load relation network success")

    print("training...")
    last_accuracy = 0.0
    last_gzsl_seen_accuracy = 0.0
    last_gzsl_unseen_accuracy = 0.0
    last_H = 0.0

    best_acc = 0.0
    for episode in range(EPISODE):
        coarse_network_scheduler.step(episode)
        attribute_network_scheduler.step(episode)
        fine_network_scheduler.step(episode)
        train_seen_f, train_seen_l = train_seen_loader.__iter__().next()
        # sample_labels = []
        # for label in train_seen_l.numpy():
        #     if label not in sample_labels:
        #         sample_labels.append(label)
        train_seen_l = torch.LongTensor([test_seen_id.tolist().index(item) for item in train_seen_l])
        train_seen_f, train_seen_l = Variable(train_seen_f.float().cuda()), Variable(train_seen_l.cuda())

        test_seen_f, test_seen_l = test_seen_loader.__iter__().next()
        test_seen_f, test_seen_l = Variable(test_seen_f.float().cuda()), Variable(test_seen_l)

        test_unseen_f, test_unseen_l = test_unseen_loader.__iter__().next()
        test_unseen_f, test_unseen_l = Variable(test_unseen_f.float().cuda()), Variable(test_unseen_l)

        test_seen_unseen_m = torch.cat((test_seen_f,test_unseen_f),0)
        # random.shuffle(test_seen_unseen_m)
        # test_seen_unseen_l = torch.cat((test_seen_l,test_unseen_l),0)
        cross = nn.CrossEntropyLoss().cuda()
        train_seen_data = coarse_network(train_seen_f)
        # test_seen_data = coarse_network(test_seen_f)
        # test_unseen_data = coarse_network(test_unseen_f)
        # seen_label = Variable(torch.ones(train_seen_data[1].shape[0])).long().cuda()
        # loss1 = cross(train_seen_data[1], seen_label)
        # seen_label = Variable(torch.ones(test_seen_data[1].shape[0])).long().cuda()
        # loss1 += cross(test_seen_data[1], seen_label)
        # unseen_label = Variable(torch.zeros(test_unseen_data[1].shape[0])).long().cuda()
        # loss1 += cross(test_unseen_data[1], unseen_label)

        # sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
        sample_attributes = torch.Tensor(att_seen_pro)
        class_num = sample_attributes.shape[0]
        sample_attributes = attribute_network(sample_attributes.cuda())

        # see_f:64*2048, seen_att_f:40*2048
        # loss_seen_dist = seendist(train_seen_f, sample_attributes, train_seen_l)
        sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(BATCH_SIZE, 1, 1)

        train_seen_m_ = train_seen_data[0].unsqueeze(0).repeat(class_num, 1, 1)
        train_seen_m_ = torch.transpose(train_seen_m_, 0, 1)
        relation_pairs = torch.cat((sample_attributes, train_seen_m_), 2).view(-1, 2048+2048)
        relations = fine_network(relation_pairs).view(-1, class_num)

        # cross = nn.CrossEntropyLoss().cuda()
        loss2 = cross(relations,Variable(train_seen_l).cuda())
# ##########################
#
        # loss3
        # sample_attributes = torch.Tensor(att_seen_pro)
        # sample_attributes = attribute_network(sample_attributes.cuda())
        # sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(test_seen_f), 1, 1)
        # test_seen_m_ = test_seen_f.unsqueeze(0).repeat(40, 1, 1)
        # seen_m_ = torch.transpose(test_seen_m_, 0, 1)
        # relation_pairs = torch.cat((sample_attributes, seen_m_), 2).view(-1, 2048 + 2048)
        # p_all = fine_network(relation_pairs).view(-1, 40)
        # p_all = F.softmax(p_all,1)
        # lamda_1 = torch.tensor(1).cuda()
        # loss3 = (-torch.log(p_all.max(1)[0])).sum() / 64.
        # loss3 = lamda_1 * loss3

        # sample_attributes = torch.Tensor(att_unseen_pro)
        # sample_attributes = attribute_network(sample_attributes.cuda())
        # sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(test_unseen_f), 1, 1)
        # test_unseen_m_ = test_unseen_f.unsqueeze(0).repeat(10, 1, 1)
        # unseen_m_ = torch.transpose(test_unseen_m_, 0, 1)
        # relation_pairs = torch.cat((sample_attributes, unseen_m_), 2).view(-1, 2048 + 2048)
        # p_all = fine_network(relation_pairs).view(-1, 10)
        # p_all = F.softmax(p_all, 1)
        # lamda_2 = torch.tensor(1).cuda()
        # loss4 = (-torch.log(p_all.max(1)[0])).sum()/64.
        # loss4 = lamda_2 * loss4

        loss = loss2
        # update
        coarse_network.zero_grad()
        attribute_network.zero_grad()
        fine_network.zero_grad()
        loss.backward()
        coarse_network_optim.step()
        attribute_network_optim.step()
        fine_network_optim.step()
        # if (episode+1)%10 == 0:
        #     print(episode + 1,"loss", loss.data.cpu().numpy().tolist(),
        #           'loss1', loss1.data.cpu().numpy().tolist(),
        #           'loss2', loss2.data.cpu().numpy().tolist())#,
                  #'loss3', loss3.data.cpu().numpy().tolist(),
                  #'loss4', loss4.data.cpu().numpy().tolist())

        if (episode+1)%100 == 0:
            # test
            # print("Testing...")

            # def compute_accuracy(data_loader,attributes_f,id):
            #     class_n = len(attributes_f)
            #     accuracy = torch.tensor(0.0).cuda()
            #     all_x = torch.tensor(0.0).cuda()
            #     for f_,l_ in data_loader:
            #         f_, l_ = Variable(f_.float().cuda()), Variable(l_.cuda())
            #         # m_ = coarse_network(f_)
            #         m_ = f_
            #         sample_attributes = torch.Tensor(attributes_f)
            #         sample_attributes = attribute_network(sample_attributes.cuda())
            #         sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(m_), 1, 1)
            #         m_ = m_.unsqueeze(0).repeat(class_n, 1, 1)
            #         m_ = torch.transpose(m_, 0, 1)
            #         relation_pairs = torch.cat((sample_attributes, m_), 2).view(-1, 2048 + 2048)
            #         relations = fine_network(relation_pairs).view(-1, class_n)
            #         p_all = F.softmax(relations, 1)
            #         preds = p_all.max(1)[1]
            #         preds = id[preds]
            #         accuracy += (preds==l_).sum()
            #         all_x += len(preds)
            #     return accuracy/1./all_x
            #
            # zsl_accuracy = compute_accuracy(test_unseen_loader,
            #                                 att_unseen_pro,
            #                                 torch.LongTensor(test_unseen_id).cuda())
            # gzsl_seen_accuracy = compute_accuracy(test_seen_loader,
            #                                       all_attributes,
            #                                       torch.LongTensor(np.arange(0,50)).cuda())
            # gzsl_unseen_accuracy = compute_accuracy(test_unseen_loader,
            #                                         all_attributes,
            #                                         torch.LongTensor(np.arange(0,50)).cuda())
            # H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

            def compute_accuracy(data_loader,attributes_f,id):
                class_n = len(attributes_f)
                accuracy = torch.tensor(0.0).cuda()
                all_x = torch.tensor(0.0).cuda()
                for f_,l_ in data_loader:
                    f_, l_ = Variable(f_.float().cuda()), Variable(l_.cuda())
                    m_ = coarse_network(f_)
                    # m_ = f_
                    sample_attributes = torch.Tensor(attributes_f)
                    sample_attributes = attribute_network(sample_attributes.cuda())
                    sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(m_[0]), 1, 1)
                    m_new = m_[0].unsqueeze(0).repeat(class_n, 1, 1)
                    m_new = torch.transpose(m_new, 0, 1)
                    relation_pairs = torch.cat((sample_attributes, m_new), 2).view(-1, 2048+2048)
                    relations = fine_network(relation_pairs).view(-1, class_n)
                    p_all = F.softmax(relations, 1)
                    preds = p_all.max(1)[1]
                    preds = id[preds]
                    accuracy += (preds==l_).sum()
                    all_x += len(preds)
                return accuracy/1./all_x

            # zsl_accuracy = compute_accuracy(test_unseen_loader,
            #                                 att_unseen_pro,
            #                                 torch.LongTensor(test_unseen_id).cuda())
            # gzsl_seen_accuracy = compute_accuracy(test_seen_loader,
            #                                       all_attributes,
            #                                       torch.LongTensor(np.arange(0,50)).cuda())
            # gzsl_unseen_accuracy = compute_accuracy(test_unseen_loader,
            #                                         all_attributes,
            #                                         torch.LongTensor(np.arange(0,50)).cuda())
            gzsl_seen_accuracy = compute_accuracy(test_seen_loader,
                                                  att_seen_pro,
                                                  torch.LongTensor(test_seen_id).cuda())
            gzsl_unseen_accuracy = compute_accuracy(test_unseen_loader,
                                                    att_unseen_pro,
                                                    torch.LongTensor(test_unseen_id).cuda())
            H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)
            # print('zsl:', zsl_accuracy)
            # print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))

            if H > last_H:
            #     # save networks
            #     torch.save(attribute_network.state_dict(), "./models/attribute_network.pkl")
            #     torch.save(fine_network.state_dict(), "./models/fine_network.pkl")
            #
            #     # print("save networks for episode:",episode)
            #
                # last_accuracy = zsl_accuracy
                (last_gzsl_seen_accuracy, last_gzsl_unseen_accuracy, last_H) = (
                gzsl_seen_accuracy, gzsl_unseen_accuracy, H)
                print('best:')
                # print('zsl:', last_accuracy)
                print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (last_gzsl_seen_accuracy, last_gzsl_unseen_accuracy, last_H))

    # np.choose()
    # random.choice()




if __name__ == '__main__':
    main()