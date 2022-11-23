import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from model import Classifier, score_rank_reg
from glob import glob
from sklearn.model_selection import train_test_split
from util import onehot, create_dataset, create_infinite_dataset, get_text_desc, construct_pair, create_pair_batch

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--data', metavar='DIR',default='',
                    help='path to dataset')
parser.add_argument('--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--seed', default=0, type=float, help='training seed')


def main():
    global args
    args = parser.parse_args()

    images = []
    labels = []

    datapath = args.data

    label_folder = [os.path.join(datapath, d) for d in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, d))]


    for i in label_folder:
        for j in glob(i+"/*", recursive = True):
            images.append(j)
            labels.append(int(i.split("/")[2][0]))

    print("Split data")
    X_train, X_val, y_train, y_val = train_test_split(images, 
                                                    labels, 
                                                    test_size=0.33,
                                                    stratify=labels,
                                                    random_state=args.seed)

    X_train = np.array(X_train)
    X_val = np.array(X_val)

    y_train = np.array([onehot(i) for i in y_train])
    y_val = np.array([onehot(i) for i in y_val])


    print("Create tensorflow dataset")
    first_error = np.where(np.argmax(y_train,axis=-1) == 0)
    second_error = np.where(np.argmax(y_train,axis=-1) == 1)
    third_error = np.where(np.argmax(y_train,axis=-1) == 2)

    first_train_dataset = create_infinite_dataset(X_train[first_error], y_train[first_error], args.batch_size)
    second_train_dataset = create_infinite_dataset(X_train[second_error], y_train[second_error], args.batch_size)
    third_train_dataset = create_dataset(X_train[third_error], y_train[third_error], args.batch_size)

    first_train_iterator = iter(first_train_dataset)
    second_train_iterator = iter(second_train_dataset)

    val_dataset = create_dataset(X_val, y_val, args.batch_size)

    # create model
    
    print("Create model")
    model = Classifier()
    X_aug_first, _ = first_train_iterator.get_next()
    X_aug_second, _ = second_train_iterator.get_next()

    model(X_aug_first, X_aug_second)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
    )

    # Data loading code

    len_train_dataset = len(X_train[third_error])//args.batch_size
    len_val_dataset = len(X_val)//args.batch_size

    train_loss = []
    val_loss = []

    print("Train model")
    for epoch in np.arange(1, (args.epochs) + 1):
    
        train_loss_logs, val_loss_logs = train_val_epoch(
            model,
            epoch,
            args.epochs,
            third_train_dataset,
            first_train_iterator,
            second_train_iterator,
            val_dataset, 
            len_train_dataset, 
            len_val_dataset)

        mean_train_loss = np.mean(train_loss_logs["total/loss"])
        mean_val_loss = np.mean(val_loss_logs["total/loss"])
        
        train_loss.append(mean_train_loss)
        val_loss.append(mean_val_loss)
        
        if mean_val_loss < min_val_loss:
            print("save weight, val loss: "+str(mean_val_loss))
            model.save_weights(args.exp_name+".h5")
            min_val_loss = mean_val_loss
        print("Finish")

@tf.function
def train_step(model, X_aug, X_aug_pair, y, y_pair, return_grad = True, training = None):
    loss_log = {
            "total/loss":     None,
            "label_self_1/loss":     None,
            "label_self_2/loss":     None,
            "label_other_1/loss":     None,
            "label_other_2/loss":     None,
            "score_rank_1/loss":     None,
            "score_rank_2/loss":     None,
        }
    
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    with tf.GradientTape() as tape:
        p_self_1, p_other_1, p_self_2, p_other_2 = model.train_pair(X_aug, X_aug_pair)

        label_loss_self_1 = cce(y,p_self_1)
        label_loss_other_1 = cce(y,p_other_1)
        label_loss_self_2 = cce(y_pair,p_self_2)
        label_loss_other_2 = cce(y_pair,p_other_2)
        
        lambda_reg = 1
        reg_1 = lambda_reg*score_rank_reg(p_other_1,p_self_1,y)
        reg_2 = lambda_reg*score_rank_reg(p_other_2,p_self_2,y_pair)       
        
        total_loss = label_loss_self_1+label_loss_other_1+label_loss_self_2+label_loss_other_2+reg_1+reg_2

        loss_log["total/loss"]  = total_loss
        loss_log["label_self_1/loss"]  = label_loss_self_1
        loss_log["label_self_2/loss"]  = label_loss_self_2
        loss_log["label_other_1/loss"]  = label_loss_other_1
        loss_log["label_other_2/loss"]  = label_loss_other_2
        loss_log["score_rank_1/loss"]  = reg_1
        loss_log["score_rank_2/loss"]  = reg_2
        
        
    if return_grad:
        gradients = tape.gradient(total_loss, model.trainable_variables)
        return loss_log, gradients
    

    return loss_log

@tf.function
def val_step(model, X_aug, y, return_grad = True, training = False):
    loss_log = {
            "total/loss":     None,
            "label_self_1/loss":     None,
        }
    
    cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    
    
    with tf.GradientTape() as tape:
        x_1 = model.pred(X_aug)
        
        label_loss_self_1 = cce(y,x_1)

        loss_log["total/loss"]  = label_loss_self_1
        loss_log["label_self_1/loss"]  = label_loss_self_1
    return loss_log, x_1, y


def train_val_classification_epoch(model, epoch, max_epoch,
    train_dataset, first_train_iterator, second_train_iterator, val_dataset, len_train_dataset, 
    len_val_dataset, epoch_iter = -1
):
    
    if train_dataset is not None:
                
        train_loss_logs = {
            "total/loss": [],
            "label_self_1/loss":[],
            "label_self_2/loss":[],
            "label_other_1/loss":[],
            "label_other_2/loss":[],
            "score_rank_1/loss":[],
            "score_rank_2/loss":[],
        }
            
        with tqdm(total=len_train_dataset, desc=get_text_desc(epoch, max_epoch, train_loss_logs, "train"), position=0) as pbar:
            
            for (X_aug, y) in iter(train_dataset):
                X_aug_first, y_first = first_train_iterator.get_next()
                X_aug_second, y_second = second_train_iterator.get_next()    
                
                X_aug_new_ord = tf.concat([X_aug, X_aug_first, X_aug_second], 0)
                y_new_ord = tf.concat([y, y_first, y_second], 0)
                
                loss_log, grads = train_classifier_step(model,
                                      X_aug_new_ord,
                                      y_new_ord,
                                      return_grad = True, training = True)
                
                
                
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                

                for k in loss_log.keys():                    
                    train_loss_logs[k].extend(loss_log[k].numpy())
                    
                    
                pbar.update()
                pbar.set_description(get_text_desc(epoch, max_epoch, train_loss_logs, "train"))
                
                
    val_loss_logs = {
        "total/loss": [],
        "label_self_1/loss":[],
        "label_self_2/loss":[],
        "label_other_1/loss":[],
        "label_other_2/loss":[],
        "score_rank_1/loss":[],
        "score_rank_2/loss":[],
    }
                
    val_classification_logs = {
        "pred": [],
        "label":[]
    }
    
    if val_dataset is not None:
        # Eval mode
        with tqdm(total=len_val_dataset, desc=get_text_desc(epoch, max_epoch, val_loss_logs, "val"), position=0) as pbar:
        
            for (X, y) in iter(val_dataset):
                
                loss_log, pred, y = val_step(model,
                                                X,  
                                                y,
                                                return_grad = False, training = False)
                
                for k in loss_log.keys():                    
                    val_loss_logs[k].extend(loss_log[k].numpy())
                
                val_classification_logs["pred"].extend(pred.numpy())
                val_classification_logs["label"].extend(y.numpy())
            
                pbar.update()
                pbar.set_description(get_text_desc(epoch, max_epoch, val_loss_logs, "val"))
                                
    return train_loss_logs, val_loss_logs, val_classification_logs

def train_val_epoch(model, epoch, max_epoch,
    train_dataset, first_train_iterator, second_train_iterator, val_dataset, len_train_dataset, 
    len_val_dataset, epoch_iter = -1
):
#     print(model)
    
    if train_dataset is not None:
                
        train_loss_logs = {
            "total/loss": [],
            "label_self_1/loss":[],
            "label_self_2/loss":[],
            "label_other_1/loss":[],
            "label_other_2/loss":[],
            "score_rank_1/loss":[],
            "score_rank_2/loss":[],
        }
            
        with tqdm(total=len_train_dataset, desc=get_text_desc(epoch, max_epoch, train_loss_logs, "train"), position=0) as pbar:
            
            for (X_aug, y) in iter(train_dataset):
                X_aug_first, y_first = first_train_iterator.get_next()
                X_aug_second, y_second = second_train_iterator.get_next()
#                 
                X_aug_embedding = model.get_embedding(X_aug)
                X_aug_first_embedding = model.get_embedding(X_aug_first)
                X_aug_second_embedding = model.get_embedding(X_aug_second)
                
                chosen_list_1 = construct_pair(X_aug_embedding, y, 
                                             X_aug_second_embedding, y_second)
        
                chosen_list_2 = construct_pair(X_aug_first_embedding, y_first, 
                                             X_aug_second_embedding, y_second)
                
                X_aug_new_ord, X_aug_pair_new_ord, y_new_ord, y_pair_new_ord = create_pair_batch(X_aug, y, 
                                                                                                 X_aug_first, y_first, 
                                                                                               X_aug_second, y_second, 
                                                                                               chosen_list_1, chosen_list_2,None)
                                     
                X_aug_new_ord = tf.stack(X_aug_new_ord)
                X_aug_pair_new_ord = tf.stack(X_aug_pair_new_ord)
                y_new_ord = tf.stack(y_new_ord)
                y_pair_new_ord = tf.stack(y_pair_new_ord)
                
                loss_log, grads = train_step(model,
                                      X_aug_new_ord,
                                      X_aug_pair_new_ord,      
                                      y_new_ord,
                                      y_pair_new_ord,
                                      return_grad = True, training = True)
                
                
                
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                

                for k in loss_log.keys():                    
                    train_loss_logs[k].extend(loss_log[k].numpy())
                    
                    
                pbar.update()
                pbar.set_description(get_text_desc(epoch, max_epoch, train_loss_logs, "train"))
                
                
    val_loss_logs = {
        "total/loss": [],
        "label_self_1/loss":[],
        "label_self_2/loss":[],
        "label_other_1/loss":[],
        "label_other_2/loss":[],
        "score_rank_1/loss":[],
        "score_rank_2/loss":[],
    }
    
    if val_dataset is not None:
        # Eval mode
        with tqdm(total=len_val_dataset, desc=get_text_desc(epoch, max_epoch, val_loss_logs, "val"), position=0) as pbar:
        
            for (X, y) in iter(val_dataset):
                
                loss_log, pred, y = val_step(model,
                                    X,
                                    y,
                                    return_grad = False, training = False)
                
                for k in loss_log.keys():                    
                    val_loss_logs[k].extend(loss_log[k].numpy())
        
                pbar.update()
                pbar.set_description(get_text_desc(epoch, max_epoch, val_loss_logs, "val"))
                                
    return train_loss_logs, val_loss_logs

if __name__ == '__main__':
    main()