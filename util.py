import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from sklearn.metrics.pairwise import euclidean_distances

from albumentations import (
    RandomBrightnessContrast, 
    Compose, 
    ShiftScaleRotate,
)


def onehot(index):
    one_vector = [0 for i in range(3)]
    if index <= 3:
        one_vector[0] = 1
    else:
        one_vector[index-3] = 1
    return np.array(one_vector,dtype=np.float32)

def read_image(image_path, image_size, label=None):
    
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_raw, channels=1)
    image = tf.image.resize(image, image_size)

    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32)
    image = image/255.0
    
    return image, label

def preprocess(image, label=None):    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    
    mean_tensor = backend.constant(-np.array(mean))
    std_tensor = backend.constant(np.array(std))
    
    image = backend.bias_add(image, mean_tensor)
    image /= std_tensor
    
    return image
    
def preprocess_aug_images(image, label):
   
    image = preprocess(image)
    
    return image, label

def read_images(image_path, label):

    image_list = read_image(image_path,(256,256))[0]
    
    return image_list, label

def augment(p=1):

    return Compose([        
            RandomBrightnessContrast(p=0.5, 
                                    brightness_limit=(-0.1,0.1),
                                    contrast_limit=(-0.1,0.2)
                                    ),
            ShiftScaleRotate(shift_limit=0.05, 
                            scale_limit=[-0.1,0.1], 
                            rotate_limit=0,
                            border_mode=0,
                            value=(0,0,0),
                            p=0.5),
            ], p=p)


def train_aug_np_func(images):
    augmentor = augment()
    img_aug = augmentor(image=images)
    img = img_aug['image']
    return img.astype(np.float32)


def image_augment(image, label):

    [image_aug] = tf.numpy_function(train_aug_np_func, [image], [tf.float32])

    
    image_aug = tf.reshape(image_aug, [256,256,3])
    
    return image_aug, label

def create_dataset(X,y,batch_size):
    train_dataset = (tf.data.Dataset.from_tensor_slices((X, y))
                 .map(read_images, num_parallel_calls=4)
                 .shuffle(2048,reshuffle_each_iteration=True)
                 .map(image_augment, num_parallel_calls=4)
                 .map(preprocess_aug_images, num_parallel_calls=4)
                 .batch(batch_size, drop_remainder=True)
                 )
    
    return train_dataset

def create_infinite_dataset(X,y,batch_size):
    train_dataset = (tf.data.Dataset.from_tensor_slices((X, y))
                 .map(read_images, num_parallel_calls=4)
                 .shuffle(2048,reshuffle_each_iteration=True)
                 .map(image_augment, num_parallel_calls=4)
                 .map(preprocess_aug_images, num_parallel_calls=4)
                 .batch(batch_size, drop_remainder=True)
                 .repeat()
                 )
    
    return train_dataset

def get_text_desc(epoch, max_epoch, loss_logs, mode = "train"):
    assert mode in ["train", "val"]
    text = f"[{epoch}/{max_epoch} - {mode}] "
    for loss_name in ["total/loss", "label_self_1/loss", "label_self_2/loss", "label_other_1/loss", "label_other_2/loss", "score_rank_1/loss", "score_rank_2/loss"]:
        filtered_loss = list(filter(lambda x: x is not None, loss_logs[loss_name]))
        if len(filtered_loss) > 0:
            text += f"{loss_name}: " + "{:.5f}, ".format(np.mean(
                filtered_loss
            ))
    text = text[:-2] if text[-2:] == ", " else text
    return text

def distance(point1, point2):
    l2_norm = euclidean_distances(point1.numpy().reshape(1, -1),point2.numpy())
    return l2_norm.reshape(-1)

def construct_pair(X_1, y_1, X_2, y_2):

    distance_list = []
    chosen_list = []
    for i in range(X_1.shape[0]):
        distance_list.append([None,None,None])
        chosen_list.append([-1,-1,-1])

        dist = distance(X_1[i],X_1)
        
        choices = np.random.choice(X_1.shape[0], 5)
        
        for j in range(X_1.shape[0]):
            if i != j:
                if distance_list[i][tf.argmax(y_1[j])] == None:
                    distance_list[i][tf.argmax(y_1[j])] = dist[j]
                    chosen_list[i][tf.argmax(y_1[j])] = j

                elif dist[j] > distance_list[i][tf.argmax(y_1[j])]:
                    distance_list[i][tf.argmax(y_1[j])] = dist[j]
                    chosen_list[i][tf.argmax(y_1[j])] = j

        dist = distance(X_1[i],X_2)
        
        choices = np.random.choice(X_2.shape[0], 5)
        
        for j in range(X_2.shape[0]):

            if distance_list[i][tf.argmax(y_2[j])] == None:
                distance_list[i][tf.argmax(y_2[j])] = dist[j]
                chosen_list[i][tf.argmax(y_2[j])] = j

            elif dist[j] < distance_list[i][tf.argmax(y_2[j])]:
                distance_list[i][tf.argmax(y_2[j])] = dist[j]
                chosen_list[i][tf.argmax(y_2[j])] = j
                
    return chosen_list

def create_pair_batch(X_1, y_1, X_2, y_2, X_3, y_3, chosen_list_1, chosen_list_2, chosen_list_3):
    
    X_aug_new_ord = []
    X_aug_pair_new_ord = []
    y_new_ord = []
    y_pair_new_ord = []
    
    X_list = [X_1, X_2, X_3]
    y_list = [y_1, y_2, y_3]
    c_list = [chosen_list_1, chosen_list_2, chosen_list_3]
    
    for i in range(len(X_list)):
        for j in range(X_list[i].shape[0]):
            for k in range(3):
                try:
                    if c_list[i] != None and c_list[i][j][k] != -1:
                        X_aug_new_ord.append(X_list[i][j])
                        X_aug_pair_new_ord.append(X_list[k][c_list[i][j][k]])
                        y_new_ord.append(y_list[i][j])
                        y_pair_new_ord.append(y_list[k][c_list[i][j][k]])
                except:
                    print(i,j,k)
                    print(c_list[i][j])
                    print(c_list[i][j][k])
                    print(X_list[k])
                    raise
                    
    return X_aug_new_ord, X_aug_pair_new_ord, y_new_ord, y_pair_new_ord

