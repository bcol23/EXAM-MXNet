import mxnet as mx
from mxnet import nd
from tqdm import tqdm


def precision_k(pred, label, k=1):
    batch_size = pred.shape[0]
    pred = pred.topk(k=k)
    
    p = 0
    for i in tqdm(range(batch_size), desc='P@' + str(k)):
        p += label[i, pred[i]].mean().as_in_context(mx.cpu()).asscalar()
    
    return p*100/batch_size


def evaluate(net, data_loader, ctx=mx.cpu()):
    p1, p3, p5 = 0, 0, 0
    
    for batch_idx, (X_batch, y_batch) in tqdm(enumerate(data_loader), 
        desc='evaluate'):

        X_batch = X_batch.as_in_context(ctx)
        y_batch = y_batch.as_in_context(ctx)

        output = net(X_batch)
        p1 += precision_k(output, y_batch, k=1)
        p3 += precision_k(output, y_batch, k=3)
        p5 += precision_k(output, y_batch, k=5)

    batches = batch_idx + 1
    p1 /= batches
    p3 /= batches
    p5 /= batches
    
    return p1, p3, p5
