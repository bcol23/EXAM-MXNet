from datetime import datetime
import glob
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, rnn, utils as gutils
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

from data import load_data
from net import EXAM
from eval import evaluate

# params
ctx = [mx.cpu()]
batch_size = 1024
X_num = -1 # num of data, -1 means all
feature_num = 50 # time step or sentence len
hidden_size = 1024
test_num = int(2e5)
epoch = 10
lr = 0.001
opt = 'adam'
if_label_embed = False # use EXAM_alter if True
if_log = True # save result
log_columns = ['loss', 'train_P@1', 'train_P@3', 'train_P@5', 'test_P@1', 'test_P@3', 'test_P@5']
log_path = './log/'
data_base_path = './'


def train(net, loss, trainer, train_data_loader, test_data_loader, epoch=200):
    log = pd.DataFrame(columns=log_columns)

    for e in tqdm(range(1, epoch + 1), desc='train'):
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_data_loader), 
            desc='train batch'):

            _batch_size=X_batch.shape[0]
            X_batch = gutils.split_and_load(X_batch, ctx, even_split=False)
            y_batch = gutils.split_and_load(y_batch, ctx, even_split=False)

            with autograd.record():
                ls = [loss(net(_X), _y)
                    for _X, _y in zip(X_batch, y_batch)]
            
            for l in ls:
                l.backward()
                train_loss += l.sum().as_in_context(mx.cpu()).asscalar() 
                
            trainer.step(batch_size=_batch_size)

        nd.waitall()
        
        train_p1, train_p3, train_p5 = evaluate(net, train_data_loader, ctx=ctx[-1])
        test_p1, test_p3, test_p5 = evaluate(net, test_data_loader, ctx=ctx[-1])

        print('\n\n\nepoch %d:\tloss %.4f' %(e, train_loss))
        print('train_p@1\t%.3f\t\ttrain_p@3\t%.3f\t\ttrain_p@5\t%.3f' %(train_p1, train_p3, train_p5))
        print('test_p@1\t%.3f\t\ttest_p@3\t%.3f\t\ttest_p@5\t%.3f' %(test_p1, test_p3, test_p5))

        if(if_log):
            _log = [[train_loss, train_p1, train_p3, train_p5, test_p1, test_p3, test_p5]]
            _log = pd.DataFrame(_log, 
                columns=log_columns)
            log = log.append(_log)
                
    if(if_log):
        return log
    else:
        return None
    

if(__name__ == '__main__'):
    if(if_label_embed):
        train_data_loader, test_data_loader, embed, label_embed = load_data(X_num=X_num, 
            feature_num=feature_num, test_num=test_num, batch_size=batch_size, 
            data_base_path=data_base_path, if_label_embed=if_label_embed, if_log=if_log)
        
        net = EXAM.EXAM_alter(feature_num, embed, label_embed, hidden_size=hidden_size)

    else:
        train_data_loader, test_data_loader, embed, label_num = load_data(X_num=X_num, 
            feature_num=feature_num, test_num=test_num, batch_size=batch_size, 
            data_base_path=data_base_path, if_label_embed=if_label_embed, if_log=if_log)
            
        net = EXAM.EXAM(feature_num, embed, label_num, hidden_size=hidden_size)

    net.initialize(init=init.Xavier(), ctx=ctx)
    net.embed.weight.set_data(embed.idx_to_vec)
    if(if_label_embed):
        net.label_embed.set_data(label_embed)

    loss = gluon.loss.SigmoidBCELoss()
    trainer = gluon.Trainer(net.collect_params(), opt, {'learning_rate': lr})

    log = train(net, loss, trainer, train_data_loader, test_data_loader, epoch=epoch)
    
    if(if_log):
        log_count = 1
        for file in glob.glob('*.csv'):
            log_count += 1
            
        with open(log_path + str(log_count) + '_params.txt', 'w') as log_params:
            log_params.write(f'batch_size = {batch_size}\nX_num = {X_num}\nfeature_num = {feature_num}\n' + 
                             f'hidden_size = {hidden_size}\ntest_num = {test_num}\nepoch = {epoch}\n' + 
                             f'lr = {lr}\nopt = {opt}\nif_label_embed = {if_label_embed}\n')
            
        if(if_label_embed):
            log.to_csv(log_path + str(log_count) + '_result-alter-' + str(datetime.now()) + '.csv', 
                       encoding='utf-8', index=False)
        else:
            log.to_csv(log_path + str(log_count) + '_result-' + str(datetime.now()) + '.csv', 
                       encoding='utf-8', index=False)
