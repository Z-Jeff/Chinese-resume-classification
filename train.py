import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    
    plotStep = []
    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []
    
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            
            #print(feature, feature.shape)
            #os._exit(1)
            with torch.no_grad():
                feature = feature.data.t() # 转置，将[W, batch] 转化为[batch, W], W为词的个数
                target = target.data.sub(1) # 因为label是1,2，所以要减一
            #feature.data.t_(), target.sub_(1)  # batch first, index align
            
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            #print(feature.shape) # [64, 43]  [batch, dim]
            
            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch, 
                                                                             steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_loss, dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                
                plotStep.append(steps)
                trainLoss.append(loss.item())
                valLoss.append(dev_loss)
                trainAcc.append(accuracy)
                valAcc.append(dev_acc)
                
                       
            elif steps % args.save_interval == 0:
                
                save(model, args.save_dir, 'snapshot', steps)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(plotStep, trainLoss, label='Training Loss')
    plt.plot(plotStep, valLoss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(plotStep, trainAcc, label='Training Acc')
    plt.plot(plotStep, valAcc, label='Validation Acc')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    jpg_file = os.path.join(args.save_dir, 'train_val.jpg')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        os.mknod(jpg_file) 
    plt.savefig(jpg_file)    
    
    log_file = os.path.join(args.save_dir, 'log.txt')
    with open(log_file, 'w+') as f:
        f.write('Iter     : ' + str(plotStep))
        f.write('Train Acc: ' + str(trainAcc))
        f.write('Val Acc:   ' + str(valAcc))
        f.write('Train Loss:' + str(trainLoss))
        f.write('Val Acc:   ' + str(valLoss))
        f.write(str(args))
    
def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        #feature.data.t_(), target.data.sub_(1)  # batch first, index align
        with torch.no_grad():
            feature = feature.data.t() # 转置，将[W, batch] 转化为[batch, W]
            target = target.data.sub(1) # 因为label是1,2，所以要减一
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return avg_loss, accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    #print(text)
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    #print(text)
    
    text = [[text_field.vocab.stoi[x] for x in text]]
    #print(text)
    #os._exit(1)
    
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    #print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

