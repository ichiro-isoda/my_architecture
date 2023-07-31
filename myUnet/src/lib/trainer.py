import os
import copy
import csv
import sys
import json
import numpy as np

import torch

class SegTrainer():

    def __init__(
            self,
            model,
            epoch,
            patchsize,
            batchsize,
            gpu,
            opbase,
            optimizer,
            mean_image=None,
            ndim=3,
            validation=True,
            iter_interval=100
    ):
        self.model = model
        self.epoch = epoch
        self.patchsize = patchsize
        self.batchsize = batchsize
        self.gpu = gpu
        self.opbase = opbase
        self.mean_image = mean_image
        self.optimizer = optimizer
        self.ndim = ndim
        self.validation = validation
        self.iter_interval = iter_interval
        self.iteration = 1

    def training(self, iterators):
        # ======================================================================
        #   this method is called when training
        #   calc acc and loss of each epoch
        #   details in each epoch is written in _trainer method (next method)
        # ======================================================================
        train_iter, val_iter = iterators
        N_train = train_iter.dataset.__len__()
        N_validation = val_iter.dataset.__len__()
        with open(self.opbase + '/result.txt', 'w') as f:
            f.write('N_train: {}\n'.format(N_train))
            f.write('N_validation: {}\n'.format(N_validation))
        acc_admin = Acc_Administrator(self.batchsize,N_train,N_validation,self.opbase)

        res = {'TP':0, 'TN':0, 'FP':0, 'FN':0, 'loss':0}
        for epoch in range(1, self.epoch + 1):
            print('[epoch {}]'.format(epoch))
            self.model.train()
            traeval, train_sum_loss, res, val_eval, validation_sum_loss = self._trainer(train_iter, val_iter,epoch, res)
            acc_admin.output_epoch_results(traeval, val_eval, train_sum_loss, validation_sum_loss, epoch)

        return acc_admin.bestScore


    def _trainer(self, train_iter, val_iter, epoch, res):
        # ===============================================
        #   this method is called in each epoch
        #   calc each iteration acc and loss
        # ===============================================
        
        # start from uneval TPs (it cause when "N_train % iter_eval != 0")
        TP = res['TP']
        TN = res['TN']
        FP = res['FP']
        FN = res['FN']
        sum_loss = res['loss']
        TP_pre, TN_pre, FP_pre, FN_pre,sum_loss_pre = 0, 0, 0, 0

        for batch in train_iter:
            x_patch, y_patch = batch
            self.optimizer.zero_grad()
            s_loss, s_output = self.model(x=x_patch.to(torch.device(self.gpu)), t=y_patch.to(torch.device(self.gpu)))
            s_loss.backward()
            self.optimizer.step()

            sum_loss += float(s_loss.to(torch.device('cpu')) * self.batchsize)

            y_patch = y_patch.to(torch.device('cpu')).numpy()[0]
            s_output = s_output.to(torch.device('cpu')).numpy()
            #make pred (0 : background, 1 : object)
            pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 1)
            countListPos = copy.deepcopy(pred.astype(np.int16) + y_patch.astype(np.int16))
            countListNeg = copy.deepcopy(pred.astype(np.int16) - y_patch.astype(np.int16))
            TP += len(np.where(countListPos.reshape(countListPos.size)==2)[0])
            TN += len(np.where(countListPos.reshape(countListPos.size)==0)[0])
            FP += len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
            FN += len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])

            if self.iteration % self.iter_interval == 0:
                # calc TPs after last evaluation (pre should be last evaluated TPs)
                tp = TP - TP_pre
                tn = TN - TN_pre
                fp = FP - FP_pre
                fn = FN - FN_pre
                loss = sum_loss - sum_loss_pre
                evals_iter = self._evaluator(tp, tn, fp, fn)
                evals_iter['loss'] = loss / self.iter_interval
                evals_iter['iteration'] = self.iteration
                with open(os.path.join(self.opbase, 'log_iter.json'), 'a') as f:
                    json.dump(evals_iter, f, indent=4)
                model_name = 'model_{0:06d}.npz'.format(self.iteration)
                if self.validation:
                    self.model.eval()
                    val_eval, validation_sum_loss = self._validator(val_iter)
                    # validation loss の処理の検討が必要　1epochの間にたくさん出てくる
                    # 精度が更新されたらモデルを保存

                else:
                    torch.save(self.model.to('cpu'), os.path.join(self.opbase, 'trained_models', model_name))
                self.model.to(torch.device(self.gpu))

                # pre should be last evaludated TPs
                TP_pre = TP
                TN_pre = TN
                FP_pre = FP
                FN_pre = FN
                sum_loss_pre = sum_loss
            
            self.iteration += 1

        # evals is "epoch" acc, so TPs_res should be subtracted because the value is from previous epoch
        evals = self._evaluator(TP-res['TP'], TN-res['TN'], FP-res['TN'], FN-res['FN'])
        epoch_loss = sum_loss - res['loss']

        # return evals and TPs which still unevaluated
        res = {'TP': TP - TP_pre, 'TN': TN - TN_pre, 'FP': FP - FP_pre, 'FN': FN - FN_pre, 'loss': sum_loss - sum_loss_pre}
        return evals, epoch_loss, res, val_eval, validation_sum_loss


    def _validator(self, dataset_iter):
        print("validation phase")
        TP, TN, FP, FN = 0, 0, 0, 0
        #dataset_iter.reset()
        sum_loss = 0
        num = 0
        for batch in dataset_iter:
            num += 1
            x_batch, y_batch = batch
            if self.ndim == 2:
                im_size = x_batch.shape[1:]
                stride = [int(self.patchsize[0]/2), int(self.patchsize[1]/2)]
                sh = [int(stride[0]/2), int(stride[1]/2)]
            elif self.ndim == 3:
                im_size = x_batch.shape[1:]
                stride = [int(self.patchsize[0]/2), int(self.patchsize[1]/2), int(self.patchsize[2]/2)]
                sh = [int(stride[0]/2), int(stride[1]/2), int(stride[2]/2)]

            ''' calculation for pad size'''
            if np.min(self.patchsize) > np.max(im_size):
                if self.ndim == 2:
                    pad_size = [self.patchsize[0], self.patchsize[1]]
                elif self.ndim == 3:
                    pad_size = [self.patchsize[0], self.patchsize[1], self.patchsize[2]]
            else:
                pad_size = []
                for axis in range(len(im_size)):
                    if (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) % stride[axis] == 0:
                        stride_num = (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) / stride[axis]
                    else:
                        stride_num = (im_size[axis] + 2*sh[axis] - self.patchsize[axis]) / stride[axis] + 1
                    pad_size.append(int(stride[axis] * stride_num + self.patchsize[axis]))

            gt = copy.deepcopy(y_batch)
            pre_img = np.zeros(pad_size)

            if self.ndim == 2:
                x_batch = mirror_extension_image(image=x_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
                y_batch = mirror_extension_image(image=y_batch, ndim=self.ndim,  length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1]]
                for y in range(0, pad_size[0]-self.patchsize[0], stride[0]):
                    for x in range(0, pad_size[1]-self.patchsize[1], stride[1]):
                        x_patch = torch.Tensor(x_batch[:, y:y+self.patchsize[0], x:x+self.patchsize[1]])
                        x_patch = x_patch.reshape(1,x_patch.shape[0],x_patch.shape[1],x_patch.shape[2])
                        y_patch = torch.Tensor(y_batch[:, y:y+self.patchsize[0], x:x+self.patchsize[1]])
                        s_loss, s_output = self.model(x=x_patch.to(torch.device(self.gpu)), t=y_patch.to(torch.device(self.gpu)), seg=False)
                        sum_loss += float(s_loss.to(torch.device('cpu')) * self.batchsize)
                        s_output = s_output.to(torch.device('cpu')).numpy()
                        pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                        # Add segmentation image
                        pre_img[y:y+stride[0], x:x+stride[1]] += pred[sh[0]:-sh[0], sh[1]:-sh[1]]
                seg_img = (pre_img > 0) * 1
                seg_img = seg_img[:im_size[0], :im_size[1]]
            elif self.ndim == 3:
                x_batch = mirror_extension_image(image=x_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
                y_batch = mirror_extension_image(image=y_batch, ndim=self.ndim, length=int(np.max(self.patchsize)))[:, self.patchsize[0]-sh[0]:self.patchsize[0]-sh[0]+pad_size[0], self.patchsize[1]-sh[1]:self.patchsize[1]-sh[1]+pad_size[1], self.patchsize[2]-sh[2]:self.patchsize[2]-sh[2]+pad_size[2]]
                for z in range(0, pad_size[0]-self.patchsize[0], stride[0]):
                    for y in range(0, pad_size[1]-self.patchsize[1], stride[1]):
                        for x in range(0, pad_size[2]-self.patchsize[2], stride[2]):
                            x_patch = torch.Tensor(np.expand_dims(x_batch[:, z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]], axis=0))
                            y_patch = torch.Tensor(y_batch[:, z:z+self.patchsize[0], y:y+self.patchsize[1], x:x+self.patchsize[2]])
                            s_loss, s_output = self.model(x=x_patch.to(torch.device(self.gpu)), t=y_patch.to(torch.device(self.gpu)), seg=False)
                            sum_loss += float(s_loss.to(torch.device('cpu')) * self.batchsize)
                            s_output = s_output.to(torch.device('cpu')).numpy()
                            pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 255)
                            # Add segmentation image
                            pre_img[z:z+stride[0], y:y+stride[1], x:x+stride[2]] += pred[sh[0]:-sh[0], sh[1]:-sh[1], sh[2]:-sh[2]]
                seg_img = (pre_img > 0) * 1
                seg_img = seg_img[:im_size[0], :im_size[1], :im_size[2]]
            gt = gt[0].numpy()
            # io.imsave('{}/segimg{}_validation.tif'.format(self.opbase, num), np.array(seg_img * 255).astype(np.uint8))
            # io.imsave('{}/gtimg{}_validation.tif'.format(self.opbase, num), np.array(gt * 255).astype(np.uint8))
            countListPos = copy.deepcopy(seg_img.astype(np.int16) + gt.astype(np.int16))
            countListNeg = copy.deepcopy(seg_img.astype(np.int16) - gt.astype(np.int16))
            TP += len(np.where(countListPos.reshape(countListPos.size)==2)[0])
            TN += len(np.where(countListPos.reshape(countListPos.size)==0)[0])
            FP += len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
            FN += len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])

        evals = self._evaluator(TP, TN, FP, FN)
        print("validation phase finished")
        return evals, sum_loss


    def _evaluator(self, TP, TN, FP, FN):

        evals = {}
        try:
            evals['Accuracy'] = (TP + TN) / float(TP + TN + FP + FN)
        except:
            evals['Accuracy'] = 0.0
        try:
            evals['Recall'] = TP / float(TP + FN)
        except:
            evals['Recall'] = 0.0
        try:
            evals['Precision'] = TP / float(TP + FP)
        except:
            evals['Precision'] = 0.0
        try:
            evals['Specificity'] = TN / float(TN + FP)
        except:
            evals['Specificity'] = 0.0
        try:
            evals['F-measure'] = 2 * evals['Recall'] * evals['Precision'] / (evals['Recall'] + evals['Precision'])
        except:
            evals['F-measure'] = 0.0
        try:
            evals['IoU'] = TP / float(TP + FP + FN)
        except:
            evals['IoU'] = 0.0
        return evals

class Acc_Administrator():
    def __init__(self,batchsize,N_train,N_validation,opbase):
        self.bestScore=0
        self.bestAccuracy=0
        self.bestRecall=0
        self.bestPrecision=0
        self.bestSpecificity=0
        self.bestFmeasure=0
        self.bestIoU=0
        self.bestEpoch=0
        self.batchsize=batchsize
        self.criteria = ['Accuracy', 'Recall', 'Precision', 'Specificity', 'F-measure', 'IoU']
        self.N_train=N_train
        self.N_validation=N_validation
        self.opbase=opbase

    def output_epoch_results(self,traeval, val_eval,train_sum_loss,validation_sum_loss,epoch):
        print('train mean loss={}'.format(train_sum_loss / (self.N_train * self.batchsize)))
        print('train accuracy={}, train recall={}'.format(traeval['Accuracy'], traeval['Recall']))
        print('train precision={}, specificity={}'.format(traeval['Precision'], traeval['Specificity']))
        print('train F-measure={}, IoU={}'.format(traeval['F-measure'], traeval['IoU']))
        print('validation mean loss={}'.format(validation_sum_loss / (self.N_validation * self.batchsize)))
        print('validation accuracy={}, recall={}'.format(val_eval['Accuracy'], val_eval['Recall']))
        print('validation precision={}, specificity={}'.format(val_eval['Precision'], val_eval['Specificity']))
        print('validation F-measure={}, IoU={}'.format(val_eval['F-measure'], val_eval['IoU']))
        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('========================================\n')
            f.write('[epoch' + str(epoch) + ']\n')
            f.write('train mean loss={}\n'.format(train_sum_loss / (self.N_train * self.batchsize)))
            f.write('train accuracy={}, train recall={}\n'.format(traeval['Accuracy'], traeval['Recall']))
            f.write('train precision={}, specificity={}\n'.format(traeval['Precision'], traeval['Specificity']))
            f.write('train F-measure={}, IoU={}\n'.format(traeval['F-measure'], traeval['IoU']))
            if epoch % self.val_epoch == 0:
                f.write('validation mean loss={}\n'.format(validation_sum_loss / (self.N_validation * self.batchsize)))
                f.write('validation accuracy={}, recall={}\n'.format(val_eval['Accuracy'], val_eval['Recall']))
                f.write('validation precision={}, specificity={}\n'.format(val_eval['Precision'], val_eval['Specificity']))
                f.write('validation F-measure={}, IoU={}\n'.format(val_eval['F-measure'], val_eval['IoU']))
        with open(self.opbase + '/TrainResult.csv', 'a') as f:
            c = csv.writer(f)
            c.writerow([epoch, traeval['Accuracy'], traeval['Recall'], traeval['Precision'], traeval['Specificity'], traeval['F-measure'], traeval['IoU']])
        with open(self.opbase + '/ValResult.csv', 'a') as f:
            c = csv.writer(f)
            c.writerow([epoch, val_eval['Accuracy'], val_eval['Recall'], val_eval['Precision'], val_eval['Specificity'], val_eval['F-measure'], val_eval['IoU']])

        if self.bestAccuracy <= val_eval['Accuracy']:
            self.bestAccuracy = val_eval['Accuracy']
        if self.bestRecall <= val_eval['Recall']:
            self.bestRecall = val_eval['Recall']
        if self.bestPrecision <= val_eval['Precision']:
            self.bestPrecision = val_eval['Precision']
        if self.bestSpecificity <= val_eval['Specificity']:
            self.bestSpecificity = val_eval['Specificity']
        if self.bestFmeasure <= val_eval['F-measure']:
            self.bestFmeasure = val_eval['F-measure']
        if self.bestIoU <= val_eval['IoU']:
            self.bestIoU = val_eval['IoU']
            self.bestEpoch = epoch
            # Save Model
            if epoch > 0:
                # epoch単位でのvalidationにおけるbestしか取ってこれない。誤解生むしこのmodel保存する必要ある？
                model_name = 'bestIoU.npz'
                torch.save(self.model.to('cpu'), os.path.join(self.opbase, model_name))
                self.model.to(torch.device(self.gpu))
            else:
                self.bestIoU = 0.0

        self.bestScore = [self.bestAccuracy, self.bestRecall, self.bestPrecision, self.bestSpecificity, self.bestFmeasure, self.bestIoU]
        print('========================================')
        print('Best Epoch : ' + str(self.bestEpoch))
        print('Best Accuracy : ' + str(self.bestAccuracy))
        print('Best Recall : ' + str(self.bestRecall))
        print('Best Precision : ' + str(self.bestPrecision))
        print('Best Specificity : ' + str(self.bestSpecificity))
        print('Best F-measure : ' + str(self.bestFmeasure))
        print('Best IoU : ' + str(self.bestIoU))
        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('################################################\n')
            f.write('BestAccuracy={}\n'.format(self.bestAccuracy))
            f.write('BestRecall={}, BestPrecision={}\n'.format(self.bestRecall, self.bestPrecision))
            f.write('BestSpecificity={}, BestFmesure={}\n'.format(self.bestSpecificity, self.bestFmeasure))
            f.write('BestIoU={}, BestEpoch={}\n'.format(self.bestIoU, self.bestEpoch))
            f.write('################################################\n')

