import os

from sklearn.model_selection import train_test_split

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import random
import time
import datetime
import scipy.io

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from torch import nn
from torch.backends import cudnn
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,\
    classification_report
import warnings
from mne.decoding import CSP
sys.path.append('../../')
from model import EEG, MEegNet, lmda, conformer, lmdaEmg, lmdaEmgCMC, MEegNetCMC
from utils import Prod_HEG_data, Prod_all_data, Prod_emg_data, Prod_cmc_data
import tSNE
# from visualization import tSNE
# 忽略所有UserWarning警告
warnings.filterwarnings("ignore", category=UserWarning)
# 设置字体为SimHei，即黑体
plt.rcParams['font.sans-serif'] = ['SimHei']

cudnn.benchmark = False
cudnn.deterministic = True



def save_confusion_matrix(test_trues_np, test_pres_np, save_name):
    # 绘制混淆矩阵并保存
    matrix = confusion_matrix(test_trues_np, test_pres_np)

    target_names = ['握拳', '屈腕', '伸腕', '屈肘']
    classes = ['握拳', '屈腕', '伸腕', '屈肘']
    print('Classification Report:')
    print(classification_report(test_trues_np, test_pres_np,
                                target_names=target_names, digits=5))

    classNamber = 4
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    plt.title('confusion_matrix')  # 改图名
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)

    thresh = matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(classNamber)] for i in range(classNamber)], (matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(matrix[i, j]), va='center', ha='center')  # 显示对应的数字

    plt.ylabel('Ture')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(save_name+'.png', dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()



class ExGAN():
    def __init__(self, nsub, datafolder, save_folder, modelname, batchsize, nepochs, d_coders,
         channels, isAug= False):
        super(ExGAN, self).__init__()
        self.batch_size = batchsize
        self.n_epochs = nepochs
        self.img_height = 22
        self.img_width = 600
        self.channels = channels
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.alpha = 0.0002
        self.dimension = (190, 50)
        self.datafolder = datafolder
        self.isAug = isAug

        self.nSub = nsub

        self.start_epoch = 0

        self.pretrain = False

        self.log_write = open(save_folder + "/restxt/log_subject%d.txt" % self.nSub, "a+")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        if modelname == 'EegNet':

            # Conformer 模型
            self.model = EEG.EegNet(chunk_size = 2500,
                     num_electrodes = self.channels,
                     in_depth = 1,
                     F1 = 16,
                     F2 = 64,
                     D = 2,
                     num_classes = 4,
                     kernel_1 = 128,
                     kernel_2 = 64,
                     dropout = 0.5).cuda()
        elif modelname == 'Eeg+Emg':
            self.model = MEegNet.MEegNet(chunk_size = 2500,
                 num_electrodes = 60).cuda()
        elif modelname.lower() == 'Eeg+Emg+CMC'.lower():
            self.model = MEegNetCMC.MEegNet(chunk_size=625,
                                         num_electrodes=channels).cuda()
        elif modelname == 'LMDA':
            # LMDA 模型
            # # c=60是因为进行了csp变换 变换后的特征数为60
            self.model = lmdaEmgCMC.LMDA(chans=60,num_classes=4,samples=2500).cuda()
        elif modelname == 'LMDA+Emg':
            self.model = lmdaEmg.LMDA(chans=60,num_classes=4,samples=2500).cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()

    def get_source_data(self, channels=44):
        data_tmp = np.load(self.datafolder + '/eegdata.npy', allow_pickle=True)
        data_tmp0 = np.load(self.datafolder + '/emgdata.npy', allow_pickle=True)
        data_cmc = np.load(self.datafolder + '/cmcdata.npy', allow_pickle=True)

        alldata_eeg, ally_eeg, traindata_eeg, trainy_eeg, testdata_eeg, testy_eeg = Prod_all_data(data_tmp, channels)
        alldata_emg, ally_emg, traindata_emg, trainy_emg, testdata_emg, testy_emg = Prod_emg_data(data_tmp0, channels)
        alldata_cmc, ally_cmc, traindata_cmc, trainy_cmc, testdata_cmc, testy_cmc = Prod_cmc_data(data_cmc, channels)

        alldata_eeg = np.array(alldata_eeg).astype(np.float64)
        alldata_emg = np.array(alldata_emg).astype(np.float64)
        alldata_cmc = np.array(alldata_cmc).astype(np.float64)[:,:,:,:]

        train_indices, test_indices = train_test_split(np.arange(len(alldata_eeg)), stratify=ally_eeg, train_size=0.8,
                                                       random_state=2024,
                                                       shuffle=True)
        traindata_eeg = alldata_eeg[train_indices]
        testdata_eeg = alldata_eeg[test_indices]
        trainy_eeg = np.array(ally_eeg)[train_indices]
        testy_eeg = np.array(ally_eeg)[test_indices]

        # 使用相同的索引切分 alldata_emg 和 ally_emg
        traindata_emg = alldata_emg[train_indices]
        testdata_emg = alldata_emg[test_indices]
        trainy_emg = np.array(ally_emg)[train_indices]
        testy_emg = np.array(ally_emg)[test_indices]

        traindata_cmc = alldata_cmc[train_indices]
        testdata_cmc = alldata_cmc[test_indices]
        trainy_cmc = np.array(ally_cmc)[train_indices]
        testy_cmc = np.array(ally_cmc)[test_indices]


        traindata_temp_eeg = np.expand_dims(np.array(traindata_eeg), axis=1)

        self.allData_eeg = traindata_temp_eeg
        self.allLabel_eeg = np.array(trainy_eeg)

        self.testData_eeg = np.expand_dims(testdata_eeg, axis=1)
        self.testLabel_eeg = np.array(testy_eeg)

        # standardize
        target_mean_eeg = np.mean(self.allData_eeg)
        target_std_eeg = np.std(self.allData_eeg)
        self.allData_eeg = (self.allData_eeg - target_mean_eeg) / target_std_eeg
        self.testData_eeg = (self.testData_eeg - target_mean_eeg) / target_std_eeg

        traindata_temp_emg = np.expand_dims(np.array(traindata_emg), axis=1)
        # shuffle_num_emg = np.random.permutation(len(traindata_temp_emg))
        self.allData_emg = traindata_temp_emg
        self.allLabel_emg = np.array(trainy_emg)

        self.testData_emg = np.expand_dims(np.array(testdata_emg), axis=1)
        self.testLabel_emg = np.array(testy_emg)

        # standardize
        target_mean_emg = np.mean(self.allData_emg)
        target_std_emg = np.std(self.allData_emg)
        self.allData_emg = (self.allData_emg - target_mean_emg) / target_std_emg
        self.testData_emg = (self.testData_emg - target_mean_emg) / target_std_emg

        traindata_temp_cmc = np.array(traindata_cmc)
        # shuffle_num_cmc = np.random.permutation(len(traindata_temp_cmc))
        self.allData_cmc = traindata_temp_cmc
        self.allLabel_cmc = np.array(trainy_cmc)

        self.testData_cmc = np.array(testdata_cmc)
        self.testLabel_cmc = np.array(testy_cmc)

        # standardize
        target_mean_cmc = np.mean(self.allData_cmc)
        target_std_cmc = np.std(self.allData_cmc)
        self.allData_cmc = (self.allData_cmc - target_mean_cmc) / target_std_cmc
        self.testData_cmc = (self.testData_cmc - target_mean_cmc) / target_std_cmc


        return self.allData_eeg, self.allLabel_eeg, self.testData_eeg, self.testLabel_eeg,\
                   self.allData_emg, self.allLabel_emg, self.testData_emg, self.testLabel_emg, \
                   self.allData_cmc, self.allLabel_cmc, self.testData_cmc, self.testLabel_cmc


    def aug(self, img, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug)
            tmp_data = img[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros(tmp_data.shape)
            tmp_aug_label = tmp_label
            for ri in range(int(len(tmp_data))):
                for rj in range(20):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 20)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = \
                        tmp_data[rand_idx[rj], :, :, rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_aug_label)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()
        aug_label = aug_label.long()

        return aug_data, aug_label

    def train(self):

        img_eeg, label_eeg, test_data_eeg, test_label_eeg, img_emg, label_emg, test_data_emg, test_label_emg,\
            img_cmc, label_cmc, test_data_cmc, test_label_cmc= self.get_source_data(self.channels)
        
        img_eeg = img_eeg.astype(np.float32)
        img_emg = img_emg.astype(np.float32)
        img_cmc = img_cmc.astype(np.float32)
        img_eeg = torch.from_numpy(img_eeg)
        label_eeg = torch.from_numpy(label_eeg)
        img_emg = torch.from_numpy(img_emg)
        img_cmc = torch.from_numpy(img_cmc)


        dataset = torch.utils.data.TensorDataset(img_eeg, img_emg, img_cmc, label_eeg)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        test_data_eeg = test_data_eeg.astype(np.float32)
        test_data_emg = test_data_emg.astype(np.float32)
        test_data_cmc = test_data_cmc.astype(np.float32)
        test_data_eeg = torch.from_numpy(test_data_eeg)
        test_data_emg = torch.from_numpy(test_data_emg)
        test_data_cmc = torch.from_numpy(test_data_cmc)
        test_label_eeg = torch.from_numpy(test_label_eeg )
        test_dataset = torch.utils.data.TensorDataset(test_data_eeg, test_data_emg, test_data_cmc, test_label_eeg)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)


        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data_eeg = Variable(test_data_eeg.type(self.Tensor))
        test_data_emg = Variable(test_data_emg.type(self.Tensor))
        test_data_cmc = Variable(test_data_cmc.type(self.Tensor))
        test_label_eeg = Variable(test_label_eeg.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        best_Feature = []

        for e in range(self.n_epochs):
            self.model.train()
            for i, (imgeeg, imgemg, imgcmc, label) in enumerate(self.dataloader):
                imgeeg = Variable(imgeeg.cuda().type(self.Tensor))
                imgemg = Variable(imgemg.cuda().type(self.Tensor))
                imgcmc = Variable(imgcmc.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))
                if str(self.isAug[:4]) == 'True':
                    aug_data, aug_label = self.aug(self.allData, self.allLabel)
                    img = torch.cat((img, aug_data))
                    label = torch.cat((label, aug_label))
                outputs,_,_,_ = self.model(imgeeg, imgemg, imgcmc)

                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if (e + 1) % 1 == 0:
                self.model.eval()
                Cls, feature_eeg, feature_emg, feature_cmc = self.model(test_data_eeg, test_data_emg, test_data_cmc)
                loss_test = self.criterion_cls(Cls, test_label_eeg)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label_eeg).cpu().numpy().astype(int).sum()) / float(test_label_eeg.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)
                # torch.cuda.empty_cache()
                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label_eeg
                    Y_pred = y_pred
                    best_Feature = [feature_eeg, feature_emg, feature_cmc]

        torch.save(self.model.module.state_dict(), '../../model.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        return bestAcc, averAcc, Y_true, Y_pred, best_Feature


def main(savefolder, datafloder, modelname='EegNet', batchsize=64, nepochs=200, depth=20, d_coders=10,
         channels=44, isAug = False):
    '''
    :param savefolder: 存储的文件夹
    :param datafloder: 数据文件夹
    :param modelname:  模型名称
    :param batchsize:  batchsize 参数设置
    :param nepochs:    模型训练次数
    :param d_coders:   LMDA的话 encoder编码层数
    :param channels:   通道数选择  是选择所有通道还是中央脑区通道
    :return:
    '''
    best = 0
    aver = 0
    if modelname != 'LMDA' and savefolder != 'EEGConformer':
        save_folder = savefolder + modelname + '_CSP_' + str(channels) + '_' + str(batchsize) + '_' + str(nepochs) + str(isAug)
    elif modelname == 'LMDA':
        save_folder = savefolder + modelname + '_CSP_' + str(channels) + '_' + str(batchsize) + '_' + str(nepochs)+ '_' + str(depth) + str(isAug)
    elif modelname == 'Conformer':
        save_folder = savefolder + modelname + '_CSP_' + str(channels) + '_' + str(batchsize) + '_' + str(
            nepochs) + '_' + str(d_coders) + str(isAug)
    # 如果没有保存结果的文件夹  就创建文件夹
    if not os.path.exists(save_folder):
        # 结果文件夹的主目录
        os.mkdir(save_folder)
        # 用来存放混淆矩阵图片的文件夹
        os.mkdir(save_folder+'/res_confusionfig')
        # 用来存放每个人模型输出结果的文件夹
        os.mkdir(save_folder + "/restxt")
    else:
        save_folder = save_folder + str(np.random.randint(0,2024))
        isAug = str(isAug) + str(np.random.randint(0,2024))
         # 结果文件夹的主目录
        os.mkdir(save_folder)
        # 用来存放混淆矩阵图片的文件夹
        os.mkdir(save_folder+'/res_confusionfig')
        # 用来存放每个人模型输出结果的文件夹
        os.mkdir(save_folder + "/restxt")
                                        
    run_sub = 4
    # sub_result.txt  用来存储所有个体的结果
    if modelname != 'LMDA' and savefolder != 'EEGConformer':
        result_write = open( savefolder + "/CSP_sub_result"+ '_' + modelname + '_' + str(channels) +
                            '_' + str(batchsize) + '_' + str(nepochs)+ '_' + str(isAug)+ '_n' + str(run_sub) +".txt", "a+")
    elif modelname == 'LMDA':
        result_write = open( savefolder + "/CSP_sub_result"+ '_' + modelname + '_' + str(channels) +
                            '_' + str(batchsize) + '_' + str(nepochs)+ '_' + str(depth) + str(isAug)+ '_n' + str(run_sub) + ".txt", "a+")
    elif modelname == 'Conformer':
        result_write = open( savefolder + "/CSP_sub_result"+ '_' + modelname + '_' + str(channels) +
                            '_' + str(batchsize) + '_' + str(nepochs)+ '_' + str(d_coders) + str(isAug)+ '_n' + str(run_sub) + ".txt", "a+")

    data_list = [os.path.join(datafloder, path) for path in os.listdir(datafloder)]
    # for i in range(len(data_list)):
    for i in range(run_sub):
        
        starttime = datetime.datetime.now()
        seed_n = 2024

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        print(data_list[i])
        exgan = ExGAN(i + 1, data_list[i], save_folder, modelname, batchsize, nepochs, d_coders,
         channels, isAug)

        bestAcc, averAcc, Y_true, Y_pred, best_Feature = exgan.train()

        save_confusion_matrix(Y_true.detach().cpu().numpy(),
                              Y_pred.detach().cpu().numpy(), save_folder+'/res_confusionfig' + '/' + str(i))

        # 绘制tsne 图
        # for feature_ind in range(len(best_Feature)):
        #     feature = best_Feature[feature_ind]
        #     tSNE.plt_tsne(feature, Y_true, 10, save_name='/'+ str(i) + '_feature' + str(feature_ind))
        torch.cuda.empty_cache()

        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('**Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
        print('当前平均精度为:', str(best/(i+1)))


    best = best / run_sub
    aver = aver / len(data_list)
    print('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    save_folder = '../../results/EEG+EMG+CMC/'
    data_folder = '../../bandwidth_each125'
    modelname = 'EEG+EMG+CMC'
    batchsize = 32
    nepochs = 200
    depth = 10
    d_coders = 10
    channels = 127
    isAug = False # 是否进行数据增强
    main(save_folder, data_folder, modelname, batchsize, nepochs, depth, d_coders, channels, isAug)
    print(time.asctime(time.localtime(time.time())))
