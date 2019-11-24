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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# class EmissionNetwork(nn.Module):
#     def __init__(self):
#         super(EmissionNetwork, self).__init__()
#         self.emission = nn.Conv1d(1, 1, 1, padding=0,stride=1, bias=False)
#         self.softmax = nn.Sigmoid()
#
#     def forward(self, input):
#         input = input.unsqueeze(1)
#         x = self.emission(input)
#         x = self.softmax(x)
#         x = x*input
#         x = x.view(x.size(0),-1)
#
#         return x

class CoarseNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(CoarseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))

        return x1, x2

class FineNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,):
        super(FineNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

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

        return x
class FeatureNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,output_size):
        super(FeatureNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

loss_fun = torch.nn.CrossEntropyLoss().cuda()


#trainData=seen, testData=seen+unseen
def trainCoarseNetWork(trainData, testData, coarse_network, loss_fun,coarse_network_optim,best_acc):
    labels = torch.from_numpy(np.concatenate([np.ones(len(trainData),dtype=int),np.zeros(len(testData),dtype=int)]))
    features = torch.from_numpy(np.concatenate([trainData,testData]))
    train_data = TensorDataset(features, labels)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=2)
    rewards = 0
    all_x = 0
    for ind, (t_f, t_l) in enumerate(train_loader):
        t_f, t_l = Variable(t_f.float().cuda()), Variable(t_l.cuda())
        preds = coarse_network(t_f)
        loss = loss_fun(preds,t_l)
        coarse_network.zero_grad()
        loss.backward()
        coarse_network_optim.step()
        _, pred = preds.max(1)
        reward = (pred == t_l).sum()
        rewards += reward
        all_x += len(t_l)
        if ind%100==0:
            print('loss',loss.data,'acc=',reward.float()/len(t_l),'t_l',len(t_l))
    print('acgacc=',rewards.float()/all_x,all_x)
    if rewards.float()/all_x>best_acc:
        best_acc = rewards.float()/all_x
        torch.save(coarse_network.state_dict(),"./models/coarse_network.pth")
    return best_acc

def main():
    # step 1: init dataset
    print("init dataset")
    
    dataroot = './data'
    dataset = 'CUB1_data'
    image_embedding = 'res101' 
    class_embedding = 'original_att'

    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
    # matcontent = sio.loadmat('vgg_features_labels_scale.mat')
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
  
    attribute = matcontent['att'].T
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
    # visual_error_list = open('datatemp_CUB_v2_error_17.txt').readlines()
    # visual_error_seen_v = eval(visual_error_list[2])
    # visual_error_unseen_v = eval(visual_error_list[3])
    visual_error_list = open('datatemp_CUB_v3_error_94.txt').readlines()
    visual_error_seen_v = eval(visual_error_list[0])
    visual_error_unseen_v = eval(visual_error_list[1])

    #train seen
    labels = torch.from_numpy(train_seen_label)
    features = torch.from_numpy(train_seen_feature)
    train_seen_data = TensorDataset(features, labels)
    train_seen_loader = DataLoader(train_seen_data, batch_size=BATCH_SIZE, shuffle=True)

    #test seen
    labels = torch.from_numpy(test_seen_label)
    features = torch.from_numpy(test_seen_feature)
    visual_error_seen = torch.from_numpy(np.array(visual_error_seen_v))
    test_seen_data = TensorDataset(features, labels, visual_error_seen)
    test_seen_loader = DataLoader(test_seen_data, batch_size=32, shuffle=True)

    # test unseen
    labels = torch.from_numpy(test_unseen_label)
    features = torch.from_numpy(test_unseen_feature)
    visual_error_unseen = torch.from_numpy(np.array(visual_error_unseen_v))
    test_unseen_data = TensorDataset(features, labels, visual_error_unseen)
    test_unseen_loader = DataLoader(test_unseen_data, batch_size=32, shuffle=True)

    # init network
    print("init networks")
    attribute_network = AttributeNetwork(312,1024,2048)
    fine_network = FineNetwork(2048+2048,1024)

    attribute_network.cuda()
    fine_network.cuda()

    attribute_network_optim = torch.optim.Adam(attribute_network.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    attribute_network_scheduler = StepLR(attribute_network_optim, step_size=200000, gamma=0.5)
    fine_network_optim = torch.optim.Adam(fine_network.parameters(),lr=LEARNING_RATE)
    fine_network_scheduler = StepLR(fine_network_optim,step_size=200000,gamma=0.5)

    if os.path.exists("./models/CUB_attribute_network.pkl"):
        attribute_network.load_state_dict(torch.load("./models/CUB_attribute_network.pkl"))
        print("load attribute network success")
    if os.path.exists("./models/CUB_fine_network.pkl"):
        fine_network.load_state_dict(torch.load("./models/CUB_fine_network.pkl"))
        print("load relation network success")

    print("training...")
    last_accuracy = 0.0
    last_gzsl_seen_accuracy = 0.0
    last_gzsl_unseen_accuracy = 0.0
    last_H = 0.0

    best_acc = 0.0
    # for episode in range(EPISODE):
    #     coarse_network_scheduler.step(episode)
    #     best_acc = trainCoarseNetWork(train_seen_feature,np.concatenate(
    #         [test_unseen_feature,test_seen_feature]),coarse_network,loss_fun,coarse_network_optim,best_acc)
    #     print('best_acc=',best_acc)
    for episode in range(EPISODE):
        attribute_network_scheduler.step(episode)
        fine_network_scheduler.step(episode)
        train_seen_f, train_seen_l = train_seen_loader.__iter__().next()
        # sample_labels = []
        # for label in train_seen_l.numpy():
        #     if label not in sample_labels:
        #         sample_labels.append(label)
        train_seen_l = torch.LongTensor([test_seen_id.tolist().index(item) for item in train_seen_l])
        train_seen_f, train_seen_l = Variable(train_seen_f.float().cuda()), Variable(train_seen_l.cuda())

        # train_seen_m, _ = coarse_network(train_seen_f)
        train_seen_m = train_seen_f
        test_seen_f, test_seen_l,_ = test_seen_loader.__iter__().next()
        test_seen_f, test_seen_l = Variable(test_seen_f.float().cuda()), Variable(test_seen_l)

        test_unseen_f, test_unseen_l,_ = test_unseen_loader.__iter__().next()
        test_unseen_f, test_unseen_l = Variable(test_unseen_f.float().cuda()), Variable(test_unseen_l)

        test_seen_unseen_m = torch.cat((test_seen_f,test_unseen_f),0)
        random.shuffle(test_seen_unseen_m)
        # test_seen_unseen_l = torch.cat((test_seen_l,test_unseen_l),0)

#         # sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
#         sample_attributes = torch.Tensor(att_seen_pro)
#         class_num = sample_attributes.shape[0]
#         sample_attributes = attribute_network(sample_attributes.cuda())
#         sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
#         train_seen_m_ = train_seen_m.unsqueeze(0).repeat(class_num, 1, 1)
#         train_seen_m_ = torch.transpose(train_seen_m_, 0, 1)
#         relation_pairs = torch.cat((sample_attributes, train_seen_m_), 2).view(-1, 2048+2048)
#         relations = fine_network(relation_pairs).view(-1, class_num)
#         # p_all = F.softmax(relations, 1)
#         # lamda_0 = torch.tensor(0.00005).cuda()
#         # loss0 = lamda_0 * -torch.log(p_all.max(1)[0]).sum()/64.
#
#         # sample_labels = np.array(sample_labels)
#         # sample_labels = train_seen_l.cpu().numpy()
#         # re_batch_labels = []
#         # for label in train_seen_l.cpu().numpy():
#         #     index = np.argwhere(sample_labels == label)
#         #     re_batch_labels.append(index[0][0])
#         # re_batch_labels = torch.LongTensor(re_batch_labels)
#
#         # loss1
#         # mse = nn.MSELoss().cuda()
#         # one_hot_labels = Variable(torch.zeros(BATCH_SIZE, class_num)
#         #                 .scatter_(1, re_batch_labels.view(-1, 1), 1)).cuda()
#         # loss1 = mse(relations, one_hot_labels)
#
#         cross = nn.CrossEntropyLoss().cuda()
#         loss1 = cross(relations,Variable(train_seen_l).cuda())
# ##########################
#
#         # #loss2
#         # if len(test_seen_unseen_m)<>0:
#         #     sample_attributes = torch.Tensor(all_attributes)
#         #     sample_attributes = attribute_network(sample_attributes.cuda())
#         #     sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(test_seen_unseen_m), 1, 1)
#         #     test_seen_unseen_m_ = test_seen_unseen_m.unsqueeze(0).repeat(150+50, 1, 1)
#         #     test_seen_unseen_m_ = torch.transpose(test_seen_unseen_m_, 0, 1)
#         #     relation_pairs = torch.cat((sample_attributes, test_seen_unseen_m_), 2).view(-1, 2048 + 2048)
#         #     relations = fine_network(relation_pairs).view(-1, 150+50)
#         #     # loss2 = cross(relations, Variable(test_seen_unseen_l).cuda())
#         #     p_all = F.softmax(relations,1)
#         #     # lamda_1 = torch.tensor(0.000001).cuda()#0.0001-0.0005
#         #     # loss2 = lamda_1*-torch.log(p_all[:,test_seen_id].sum(1)).sum()/64.
#         #     lamda_2 = torch.tensor(0.0000005).cuda()
#         #     loss3 = (lamda_2 * -torch.log(p_all[:, test_unseen_id].sum(1))).sum()/64.
#         #     # loss2 += lamda_1 * -torch.log(p_all[:, test_seen_id].max(1)[0]).sum()/64.
#         #     # loss3 += lamda_2 * -torch.log(p_all[:, test_unseen_id].max(1)[0]).sum()/64.
#         # else:
#         #     loss2 = torch.tensor(0.).cuda()
#         #     loss3 = torch.tensor(0.).cuda()
#         #
#         # # # loss3
#         # if len(test_seen_unseen_m) <> 0:
#         #     sample_attributes = torch.Tensor(att_seen_pro)
#         #     sample_attributes = attribute_network(sample_attributes.cuda())
#         #     sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(test_seen_unseen_m), 1, 1)
#         #     test_seen_unseen_m_ = test_seen_unseen_m.unsqueeze(0).repeat(len(att_seen_pro), 1, 1)
#         #     test_seen_unseen_m_ = torch.transpose(test_seen_unseen_m_, 0, 1)
#         #     relation_pairs = torch.cat((sample_attributes, test_seen_unseen_m_), 2).view(-1, 2048 + 2048)
#         #     relations = fine_network(relation_pairs).view(-1, len(att_seen_pro))
#         #     p_all = F.softmax(relations, 1)
#         #     lamda_4 = torch.tensor(0.000001).cuda()
#         #     loss2 = lamda_4 * -torch.log(p_all.max(1)[0]).sum()/64.
#         # if len(test_seen_unseen_m) <> 0:
#         #     sample_attributes = torch.Tensor(att_unseen_pro)
#         #     sample_attributes = attribute_network(sample_attributes.cuda())
#         #     sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(test_seen_unseen_m), 1, 1)
#         #     test_seen_unseen_m_ = test_seen_unseen_m.unsqueeze(0).repeat(len(att_unseen_pro), 1, 1)
#         #     test_seen_unseen_m_ = torch.transpose(test_seen_unseen_m_, 0, 1)
#         #     relation_pairs = torch.cat((sample_attributes, test_seen_unseen_m_), 2).view(-1, 2048 + 2048)
#         #     relations = fine_network(relation_pairs).view(-1, len(att_unseen_pro))
#         #     p_all = F.softmax(relations, 1)
#         #     lamda_5 = torch.tensor(0.000001).cuda()
#         #     loss3 += lamda_5 * -torch.log(p_all.max(1)[0]).sum()/64.
#
#         #loss4
#         lamda_3 = torch.tensor(0.00001).cuda()
#         l2_reg = torch.tensor(0.).cuda()
#         for param in attribute_network.parameters():
#             l2_reg += torch.norm(param)
#         for param in fine_network.parameters():
#             l2_reg += torch.norm(param)
#         loss4 = lamda_3 * l2_reg
#         loss = loss1  + loss4
#         # update
#         attribute_network.zero_grad()
#         fine_network.zero_grad()
#         loss.backward()
#         attribute_network_optim.step()
#         fine_network_optim.step()
#         if (episode+1)%100 == 0:
#             print(episode + 1,"loss", loss.data.cpu().numpy().tolist(),
#                   'loss1', loss1.data.cpu().numpy().tolist(),
#                   # 'loss0', loss0.data.cpu().numpy().tolist(),  # train seen
#                   # 'loss2', loss2.data.cpu().numpy().tolist(),  # seen
#                   # 'loss3', loss3.data.cpu().numpy().tolist(),  # unseen
#                   'loss4', loss4.data.cpu().numpy().tolist())

        if (episode+1)%1 == 0:
            # test
            print("Testing...")

            def compute_accuracy(data_loader,attributes_f,id):
                class_n = len(attributes_f)
                accuracy = torch.tensor(0.0).cuda()
                all_x = torch.tensor(0.0).cuda()
                for f_,l_,visual_info in data_loader:
                    f_, l_ = Variable(f_.float().cuda()), Variable(l_.cuda())
                    # m_, _ = coarse_network(f_)
                    m_ = f_
                    sample_attributes = torch.Tensor(attributes_f)
                    sample_attributes = attribute_network(sample_attributes.cuda())
                    sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(m_), 1, 1)
                    m_ = m_.unsqueeze(0).repeat(class_n, 1, 1)
                    m_ = torch.transpose(m_, 0, 1)
                    relation_pairs = torch.cat((sample_attributes, m_), 2).view(-1, 2048 + 2048)
                    relations = fine_network(relation_pairs).view(-1, class_n)
                    p_all = F.softmax(relations, 1)
                    preds = p_all.max(1)[1]
                    preds = id[preds]
                    accuracy += (preds==l_).sum()
                    all_x += len(preds)
                return accuracy/1./all_x

            zsl_accuracy = compute_accuracy(test_unseen_loader,
                                            att_unseen_pro,
                                            torch.LongTensor(test_unseen_id).cuda())
            gzsl_seen_accuracy = compute_accuracy(test_seen_loader,
                                                  all_attributes,
                                                  torch.LongTensor(np.arange(0,150+50)).cuda())
            gzsl_unseen_accuracy = compute_accuracy(test_unseen_loader,
                                                    all_attributes,
                                                    torch.LongTensor(np.arange(0,150+50)).cuda())
            H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

            print('zsl:', zsl_accuracy)
            print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
            # f = open('result05.txt', 'a')
            # f.write(str(zsl_accuracy.data.cpu().numpy()) + ',' + str(gzsl_seen_accuracy.data.cpu().numpy()) + ','
            #         + str(gzsl_unseen_accuracy.data.cpu().numpy()) + ',' + str(H.data.cpu().numpy()) + '\n')
            # f.close()

            def compute_accuracy2(data_loader,attributes_f,id):
                class_n = len(attributes_f)
                accuracy = torch.tensor(0.0).cuda()
                all_x = torch.tensor(0.0).cuda()
                for f_,l_,visual_info in data_loader:
                    f_, l_ = Variable(f_.float().cuda()), Variable(l_.cuda())
                    # m_ = coarse_network(f_)
                    if class_n==150:
                        m_ = f_[visual_info<=0.07]
                        l_ = l_[visual_info<=0.07]
                    elif class_n==50:
                        m_ = f_[visual_info>0.07]
                        l_ = l_[visual_info>0.07]
                    sample_attributes = torch.Tensor(attributes_f)
                    sample_attributes = attribute_network(sample_attributes.cuda())
                    sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(m_), 1, 1)
                    m_ = m_.unsqueeze(0).repeat(class_n, 1, 1)
                    m_ = torch.transpose(m_, 0, 1)
                    relation_pairs = torch.cat((sample_attributes, m_), 2).view(-1, 2048 + 2048)
                    relations = fine_network(relation_pairs).view(-1, class_n)
                    p_all = F.softmax(relations, 1)
                    preds = p_all.max(1)[1]
                    preds = id[preds]
                    accuracy += (preds==l_).sum()
                    # all_x += len(preds)
                    all_x += len(f_)
                return accuracy/1./all_x

            gzsl_seen_accuracy = compute_accuracy2(test_seen_loader,
                                                   att_seen_pro,
                                                   torch.LongTensor(test_seen_id).cuda())
            gzsl_unseen_accuracy = compute_accuracy2(test_unseen_loader,
                                                     att_unseen_pro,
                                                     torch.LongTensor(test_unseen_id).cuda())
            H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

            print('zsl:', zsl_accuracy)
            print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))

            def compute_accuracy3(data_loader,attributes_f,id,attributes_f_2,id_2):
                class_n = len(attributes_f)
                class_n_all = len(attributes_f_2)
                accuracy = torch.tensor(0.0).cuda()
                all_x = torch.tensor(0.0).cuda()
                for f_,l_,visual_info in data_loader:
                    f_, l_ = Variable(f_.float().cuda()), Variable(l_.cuda())
                    # m_ = coarse_network(f_)
                    if class_n==150:
                        m_1 = f_[visual_info<=0.05]
                        l_1 = l_[visual_info<=0.05]
                    elif class_n==50:
                        m_1 = f_[visual_info>0.09]
                        l_1 = l_[visual_info>0.09]
                    sample_attributes = torch.Tensor(attributes_f)
                    sample_attributes = attribute_network(sample_attributes.cuda())
                    sample_attributes = sample_attributes.cuda().unsqueeze(0).repeat(len(m_1), 1, 1)
                    m_1 = m_1.unsqueeze(0).repeat(class_n, 1, 1)
                    m_1 = torch.transpose(m_1, 0, 1)
                    relation_pairs = torch.cat((sample_attributes, m_1), 2).view(-1, 2048 + 2048)
                    relations = fine_network(relation_pairs).view(-1, class_n)
                    p_all = F.softmax(relations, 1)
                    preds = p_all.max(1)[1]
                    preds = id[preds]
                    accuracy += (preds==l_1).sum()
                    # all_x += len(preds)
                    if class_n_all==200:
                        left = visual_info <= 0.09
                        right = visual_info > 0.05
                        kan = left * right
                        m_2 = f_[kan]
                        l_2 = l_[kan]
                    if len(m_2)<>0:
                        sample_attributes_2 = torch.Tensor(attributes_f_2)
                        sample_attributes_2 = attribute_network(sample_attributes_2.cuda())
                        sample_attributes_2 = sample_attributes_2.cuda().unsqueeze(0).repeat(len(m_2), 1, 1)
                        m_2 = m_2.unsqueeze(0).repeat(class_n_all, 1, 1)
                        m_2 = torch.transpose(m_2, 0, 1)
                        relation_pairs = torch.cat((sample_attributes_2, m_2), 2).view(-1, 2048 + 2048)
                        relations = fine_network(relation_pairs).view(-1, class_n_all)
                        p_all = F.softmax(relations, 1)
                        preds = p_all.max(1)[1]
                        preds = id_2[preds]
                        accuracy += (preds == l_2).sum()
                    all_x += len(f_)
                return accuracy/1./all_x

            gzsl_seen_accuracy = compute_accuracy3(test_seen_loader,
                                                   att_seen_pro,
                                                   torch.LongTensor(test_seen_id).cuda(),
                                                   all_attributes,
                                                   torch.LongTensor(np.arange(0, 200)).cuda())
            gzsl_unseen_accuracy = compute_accuracy3(test_unseen_loader,
                                                     att_unseen_pro,
                                                     torch.LongTensor(test_unseen_id).cuda(),
                                                   all_attributes,
                                                   torch.LongTensor(np.arange(0, 200)).cuda())
            H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)

            print('zsl:', zsl_accuracy)
            print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
            # if H > last_H:
            #     # save networks
            #     torch.save(attribute_network.state_dict(), "./models/CUB_attribute_network.pkl")
            #     torch.save(fine_network.state_dict(), "./models/CUB_fine_network.pkl")
            #
            #     # print("save networks for episode:",episode)
            #
            #     # last_accuracy = zsl_accuracy
            #     (last_gzsl_seen_accuracy, last_gzsl_unseen_accuracy, last_H) = (
            #     gzsl_seen_accuracy, gzsl_unseen_accuracy, H)
            print('best:')
            # print('zsl:', last_accuracy)
            print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (last_gzsl_seen_accuracy, last_gzsl_unseen_accuracy, last_H))

    # np.choose()
    # random.choice()




if __name__ == '__main__':
    main()