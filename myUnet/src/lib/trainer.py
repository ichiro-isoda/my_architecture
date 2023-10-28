import os
import copy
import csv
import sys
import json
import numpy as np

import torch
from src.lib.sliding_window import sliding_window

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
            iter_interval=100,
            test_style='sliding_window'
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
        self.test_style = test_style

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
            traeval, train_sum_loss, res = self._trainer(train_iter, val_iter,epoch, res, acc_admin)
            acc_admin.output_epoch_results(traeval, train_sum_loss, epoch)

        return acc_admin.bestScore


    def _trainer(self, train_iter, val_iter, epoch, res, acc_admin):
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
        TP_pre, TN_pre, FP_pre, FN_pre,sum_loss_pre = 0, 0, 0, 0, 0

        for batch in train_iter:
            x_patch, y_patch = batch
            self.optimizer.zero_grad()
            self.model.train()
            if self.test_style!='sliding_window':
                shrink = 2**self.model.depth
                if self.ndim == 3:
                    ch, z, y, x = x_patch.shape
                    z_pd = 0 if z%shrink == 0 else shrink-z%shrink
                    y_pd = 0 if y%shrink == 0 else shrink-y%shrink
                    x_pd = 0 if x%shrink == 0 else shrink-x%shrink
                    x_patch = torch.tensor(np.expand_dims(np.expand_dims(np.pad(x_patch[0],((z_pd,0),(y_pd,0),(x_pd,0)),mode='edge'),0),0))
                    y_patch = torch.tensor(np.expand_dims(np.pad(y_patch[0],((z_pd,0),(y_pd,0),(x_pd,0)),mode='edge'),0))
                else:
                    ch, y, x = x_patch.shape
                    y_pd = 0 if y%shrink == 0 else shrink-y%shrink
                    x_pd = 0 if x%shrink == 0 else shrink-x%shrink
                    x_patch = torch.tensor(np.expand_dims(np.expand_dims(np.pad(x_patch[0],((y_pd,0),(x_pd,0)),mode='edge'),0),0))
                    y_patch = torch.tensor(np.expand_dims(np.pad(y_patch[0],((y_pd,0),(x_pd,0)),mode='edge'),0))
                    
            s_loss, s_output = self.model(x=x_patch.to(torch.device(self.gpu)), t=y_patch.to(torch.device(self.gpu)))
            s_loss.backward()
            self.optimizer.step()

            sum_loss += float(s_loss.to(torch.device('cpu')) * self.batchsize)
            del s_loss
            y_patch = y_patch.to(torch.device('cpu')).numpy()[0]
            s_output = s_output.numpy()
            #make pred (0 : background, 1 : object)
            pred = [copy.deepcopy((0 < (s_output[b][1] - s_output[b][0])) * 1) for b in range(self.batchsize)]
            if self.test_type != 'sliding_window':
                if self.ndim==2:
                    pred = pred[:,y_pd:,x_pd:]
                    y_patch = y_patch[:,y_pd:, x_pd:]
                else:
                    pred = pred[:, z_pd:, y_pd:, x_pd:]
                    y_patch = y_patch[:, z_pd:, x_pd:]
            for b in range(self.batchsize):
                countListPos = copy.deepcopy(pred[b].astype(np.int16) + y_patch[b].astype(np.int16))
                countListNeg = copy.deepcopy(pred[b].astype(np.int16) - y_patch[b].astype(np.int16))
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
                    val_result = acc_admin.output_validation(val_eval, validation_sum_loss)
                    # 精度が更新されたらモデルを保存
                    if val_result == 1:
                        model_name = 'bestIoU.npz'
                        torch.save(self.model.to('cpu'), os.path.join(self.opbase, model_name))
                        self.model.to(torch.device(self.gpu))

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
        return evals, epoch_loss, res


    def _validator(self, dataset_iter):
        print("validation phase")
        TP, TN, FP, FN = 0, 0, 0, 0
        #dataset_iter.reset()
        sum_loss = 0
        num = 0
        for batch in dataset_iter:
            num += 1
            x_batch, y_batch = batch
            if self.test_style=='sliding_window':
                loss, pred = sliding_window(x_batch,y_batch,self.ndim,self.patchsize,self.model,self.gpu,self.resolution,loss=True)
                pred = np.expand_dims(pred,0)

            else:
                shrink = 2**self.model.depth
                if self.ndim == 3:
                    ch, z, y, x = x_patch.shape
                    z_pd = 0 if z%shrink == 0 else shrink-z%shrink
                    y_pd = 0 if y%shrink == 0 else shrink-y%shrink
                    x_pd = 0 if x%shrink == 0 else shrink-x%shrink
                    x_patch = torch.tensor(np.expand_dims(np.expand_dims(np.pad(x_patch[0],((z_pd,0),(y_pd,0),(x_pd,0)),mode='edge'),0),0))
                    y_patch = torch.tensor(np.expand_dims(np.pad(y_patch[0],((z_pd,0),(y_pd,0),(x_pd,0)),mode='edge'),0))
                else:
                    ch, y, x = x_patch.shape
                    y_pd = 0 if y%shrink == 0 else shrink-y%shrink
                    x_pd = 0 if x%shrink == 0 else shrink-x%shrink
                    x_patch = torch.tensor(np.expand_dims(np.expand_dims(np.pad(x_patch[0],((y_pd,0),(x_pd,0)),mode='edge'),0),0))
                    y_patch = torch.tensor(np.expand_dims(np.pad(y_patch[0],((y_pd,0),(x_pd,0)),mode='edge'),0))
                loss, s_output = self.model(x_batch,y_batch)
                pred = copy.deepcopy((0 < (s_output[0][1] - s_output[0][0])) * 1)
                if self.ndim==2:
                    pred = pred[:,y_pd:,x_pd:]
                    y_patch = y_patch[:,y_pd:, x_pd:]
                else:
                    pred = pred[:, z_pd:, y_pd:, x_pd:]
                    y_patch = y_patch[:, z_pd:, x_pd:]
            gt = y_batch.numpy()
            # io.imsave('{}/segimg{}_validation.tif'.format(self.opbase, num), np.array(seg_img * 255).astype(np.uint8))
            # io.imsave('{}/gtimg{}_validation.tif'.format(self.opbase, num), np.array(gt * 255).astype(np.uint8))
            for b in self.batchsize:
                countListPos = copy.deepcopy(pred[b].astype(np.int16) + gt[b].astype(np.int16))
                countListNeg = copy.deepcopy(pred[b].astype(np.int16) - gt[b].astype(np.int16))
                TP += len(np.where(countListPos.reshape(countListPos.size)==2)[0])
                TN += len(np.where(countListPos.reshape(countListPos.size)==0)[0])
                FP += len(np.where(countListNeg.reshape(countListNeg.size)==1)[0])
                FN += len(np.where(countListNeg.reshape(countListNeg.size)==-1)[0])
            sum_loss+=loss
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

    def output_epoch_results(self,traeval, train_sum_loss,epoch):
        print('train mean loss={}'.format(train_sum_loss / (self.N_train * self.batchsize)))
        print('train accuracy={}, train recall={}'.format(traeval['Accuracy'], traeval['Recall']))
        print('train precision={}, specificity={}'.format(traeval['Precision'], traeval['Specificity']))
        print('train F-measure={}, IoU={}'.format(traeval['F-measure'], traeval['IoU']))
        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('========================================\n')
            f.write('[epoch' + str(epoch) + ']\n')
            f.write('train mean loss={}\n'.format(train_sum_loss / (self.N_train * self.batchsize)))
            f.write('train accuracy={}, train recall={}\n'.format(traeval['Accuracy'], traeval['Recall']))
            f.write('train precision={}, specificity={}\n'.format(traeval['Precision'], traeval['Specificity']))
            f.write('train F-measure={}, IoU={}\n'.format(traeval['F-measure'], traeval['IoU']))
        with open(self.opbase + '/TrainResult.csv', 'a') as f:
            c = csv.writer(f)
            c.writerow([epoch, traeval['Accuracy'], traeval['Recall'], traeval['Precision'], traeval['Specificity'], traeval['F-measure'], traeval['IoU']])

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


    def output_validation(self, val_eval, validation_sum_loss):
        print('validation mean loss={}'.format(validation_sum_loss / (self.N_validation * self.batchsize)))
        print('validation accuracy={}, recall={}'.format(val_eval['Accuracy'], val_eval['Recall']))
        print('validation precision={}, specificity={}'.format(val_eval['Precision'], val_eval['Specificity']))
        print('validation F-measure={}, IoU={}'.format(val_eval['F-measure'], val_eval['IoU']))
        with open(self.opbase + '/result.txt', 'a') as f:
            f.write('validation mean loss={}\n'.format(validation_sum_loss / (self.N_validation * self.batchsize)))
            f.write('validation accuracy={}, recall={}\n'.format(val_eval['Accuracy'], val_eval['Recall']))
            f.write('validation precision={}, specificity={}\n'.format(val_eval['Precision'], val_eval['Specificity']))
            f.write('validation F-measure={}, IoU={}\n'.format(val_eval['F-measure'], val_eval['IoU']))
        with open(self.opbase + '/ValResult.csv', 'a') as f:
            c = csv.writer(f)
            c.writerow([val_eval['Accuracy'], val_eval['Recall'], val_eval['Precision'], val_eval['Specificity'], val_eval['F-measure'], val_eval['IoU']])
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
            return 1 # Save Model
        else:
            return 0
