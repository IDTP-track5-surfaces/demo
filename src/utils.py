from __future__ import print_function



from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.losses import *
import tensorflow as tf

import tensorflow.keras.backend as K

import os
import shutil
import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import pandas as pd
# Keras losses




def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def snell_refraction(normal,s1,n1,n2):
    this_normal = normal
    term_1 = tf.cross(this_normal,tf.cross(-this_normal,s1))
    term_temp = tf.cross(this_normal,s1)   
    n_sq = (n1/n2)**2
    term_2 = tf.sqrt(tf.subtract(1.0,tf.multiply(n_sq,tf.reduce_sum(tf.multiply(term_temp,term_temp),axis = 3))))   
    term_3 = tf.stack([term_2, term_2, term_2],axis = 3)   
    nn = (n1/n2)
    s2 = tf.subtract(tf.multiply(nn,term_1) , tf.multiply(this_normal,term_3))
    return s2    

# Custom loss
def depth_loss (y_true, y_pred): 
    d = tf.subtract(y_pred,y_true)
    n_pixels = 128 * 128
    square_n_pixels = n_pixels * n_pixels
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d,1)
    sum_d = tf.reduce_sum(d,1)
    square_sum_d = tf.square(sum_d)
    mid_output = tf.reduce_mean((sum_square_d/n_pixels) - 0.5* (square_sum_d/square_n_pixels))

    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    paddings_y = tf.constant([[0,0],[1,0],[0,0],[0,0]])
    paddings_x = tf.constant([[0,0],[0,0],[1,0],[0,0]])
    
    pad_dy_true = tf.pad(dy_true, paddings_y, "CONSTANT")
    pad_dy_pred = tf.pad(dy_pred, paddings_y, "CONSTANT")
    pad_dx_true = tf.pad(dx_true, paddings_x, "CONSTANT")
    pad_dx_pred = tf.pad(dx_pred, paddings_x, "CONSTANT")

    pad_dy_true = pad_dy_true[:,:-1,:,:]
    pad_dy_pred = pad_dy_pred[:,:-1,:,:]
    pad_dx_true = pad_dx_true[:,:,:-1,:]
    pad_dx_pred = pad_dx_pred[:,:,:-1,:]

    term3 = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true) + K.abs(pad_dy_pred - pad_dy_true) + K.abs(pad_dx_pred - pad_dx_true), axis=-1)
    
    depth_output = mid_output + term3
    depth_output = K.mean(depth_output)
    return depth_output

def normal_loss(y_true, y_pred):
    d = tf.subtract(y_pred,y_true)
    n_pixels = 128 * 128
    square_n_pixels = n_pixels * n_pixels
    square_d = tf.square(d)
    sum_square_d = tf.reduce_sum(square_d,1)
    sum_d = tf.reduce_sum(d,1)
    square_sum_d = tf.square(sum_d)
    normal_output = tf.reduce_mean((sum_square_d/n_pixels) - 0.5* (square_sum_d/square_n_pixels))
    return normal_output

def depth_to_normal(y_pred_depth, y_true_normal, scale_pred, scale_true):
    Scale = 127.5

    depth_min = scale_pred[:, 0:1, 0:1, :]
    depth_max = scale_pred[:, 0:1, 1:2, :]
    
    normal_min = scale_true[:, 0:1, 2:3, :]
    normal_max = scale_true[:, 0:1, 3:4, :]

    # Expand 'depth_min' and 'depth_max' to the same height and width as 'y_pred_depth'
    depth_min = tf.tile(depth_min, [1, 128, 128, 1])
    depth_max = tf.tile(depth_max, [1, 128, 128, 1])

    # Rescale 'y_pred_depth' using 'depth_min' and 'depth_max'
    y_pred_depth = depth_min + (depth_max - depth_min) * y_pred_depth

    # Expand 'normal_min' and 'normal_max' to the same height, width and channels as 'y_true_normal'
    normal_min = tf.tile(normal_min, [1, 128, 128, 3])
    normal_max = tf.tile(normal_max, [1, 128, 128, 3])

    # Rescale 'y_true_normal' using 'normal_min' and 'normal_max'
    y_true_normal = normal_min + (normal_max - normal_min) * y_true_normal
    
    # Calculate gradients
    zy, zx = tf.image.image_gradients(y_pred_depth)
    zx = zx * Scale
    zy = zy * Scale
    
    # Calculate the normal from depth
    normal_ori = tf.concat([zy, -zx, tf.ones_like(y_pred_depth)], 3)
    new_normal = tf.sqrt(tf.square(zx) +  tf.square(zy) + 1)
    normal = normal_ori / new_normal
    
    # Normalize the normals to be between 0 and 1
    normal = (normal + 1) / 2
    
    return normal, y_true_normal

def vae_loss( y_true, y_pred):
    loss1 = mean_squared_error(y_true, y_pred)
    loss2 = BinaryCrossentropy(y_true, y_pred)
    return tf.reduce_mean(loss1 + loss2)

def scale_loss(y_true,y_pred):
    pred_depth_min = y_pred[:,0:1,0:1]
    pred_depth_max = y_pred[:,0:1,1:2]
    pred_normal_min = y_pred[:,0:1,2:3]  
    pred_normal_max = y_pred[:,0:1,3:4]

    true_depth_min = y_true[:,0:1,0:1]
    true_depth_max = y_true[:,0:1,1:2]
    true_normal_min = y_true[:,0:1,2:3]  
    true_normal_max = y_true[:,0:1,3:4]

    loss_depth_min = mean_squared_error(true_depth_min, pred_depth_min)
    loss_depth_max = mean_squared_error(true_depth_max, pred_depth_max)
    loss_normal_min = mean_squared_error(true_normal_min, pred_normal_min)
    loss_normal_max = mean_squared_error(true_normal_max, pred_normal_max)

    return tf.reduce_mean(loss_depth_min + loss_depth_max + loss_normal_min + loss_normal_max)

def raytracing_loss(depth,normal,background,scale): 
    step = 255/2
    n1 = 1
    n2 = 1.33
      
    depth_min = scale[:,0:1,0:1]
    depth_max = scale[:,0:1,1:2]
    normal_min = scale[:,0:1,2:3]  
    normal_max = scale[:,0:1,3:4]

    depth = depth_min + (depth_max - depth_min) * depth
    normal = normal_min + (normal_max - normal_min) * normal

    depth = tf.squeeze(depth, axis = -1)

    s1 = tf.Variable(0.0,name="s1")
    s1_temp = tf.zeros([K.shape(depth)[0],128,128,1])
    s1 = tf.assign(s1,s1_temp, validate_shape=False)
            
    s11 = tf.Variable(0.0,name="s11")
    s11_temp = -1*tf.ones([K.shape(depth)[0],128,128,1])
    s11 = tf.assign(s11,s11_temp, validate_shape=False)

    assigned_s1 = tf.stack([s1,s1,s11],axis = 3)
    assigned_s1 = tf.squeeze(assigned_s1)

    s2 = snell_refraction(normal,assigned_s1,n1,n2) 

    x_c_ori, y_c_ori, lamda_ori = tf.split(s2,[1,1,1],axis = 3)

    lamda = -1*tf.divide(depth,tf.squeeze(lamda_ori))
    
    x_c = tf.multiply(lamda , tf.squeeze(x_c_ori))*step
    y_c = tf.multiply(lamda , tf.squeeze(y_c_ori))*step   

    flow = tf.stack([y_c,-x_c],axis = -1)

    out_im_temp = tf.contrib.image.dense_image_warp(
                    background,
                    flow,
                    name='dense_image_warp'
                    )

    out_im_tensor = tf.Variable(0.0)

    out_im_tensor = tf.assign(out_im_tensor, out_im_temp, validate_shape=False)
    return out_im_tensor

def combined_loss(y_true,y_pred):
    #print(K.int_shape(y_true)[0],K.shape(y_pred))
    print(y_true.shape)
    print(y_pred.shape)

    depth_true = y_true[:,:,:,0]
    normal_true = y_true[:,:,:,1:4]
    img_true = y_true[:,:,:,4:7]
    ref_true = y_true[:,:,:,7:10]
    scale_true = y_true[:,:,:,10:]

    depth_pred = y_pred[:,:,:,0]
    normal_pred = y_pred[:,:,:,1:4]
    img_pred = y_pred[:,:,:,4:7]
    ref_pred = y_pred[:,:,:,7:10]
    scale_pred = y_pred[:,:,:,10:]

    depth_true = tf.expand_dims(depth_true, -1)
    depth_pred = tf.expand_dims(depth_pred, -1)

    alpha = 0.2
    beta = 0.2
    gamma = 0.2
    delta = 0.2
    theta = 0.2
    tau = 0.0
    lamda = 0

    #depth loss
    loss_depth = alpha*(depth_loss(depth_true,depth_pred))

    #normal loss
    loss_normal = beta*(normal_loss(normal_true,normal_pred))
    
    #normal from depth
    # normal_from_depth, rescaled_true_normal = depth_to_normal(depth_pred,normal_true,scale_pred,scale_true)
    # loss_depth_to_normal = gamma*(normal_loss(rescaled_true_normal,normal_from_depth)) 

    #ray_tracing
    ray_traced_tensor= raytracing_loss(depth_pred,normal_pred,ref_true,scale_true)
    loss_ray_trace = delta * vae_loss(img_true,ray_traced_tensor)

    #scale_loss
    loss_scale = theta * scale_loss(scale_true,scale_pred)

    #reference_loss
    loss_reference = tau * vae_loss(ref_true,ref_pred)

    #input image loss
    loss_in_img = lamda * vae_loss(img_true,img_pred)

    return (loss_depth + loss_normal + loss_ray_trace + loss_scale + loss_reference + loss_in_img)

def move_file(file_path, dest_dir):
    if os.path.exists(file_path):
        dest_path = os.path.join(dest_dir, os.path.basename(file_path))
        shutil.move(file_path, dest_path)

def plot_metrics(history,date_time):
    # Plot training & validation loss values
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot each of the other metrics
    metrics = [key for key in history.history.keys() if 'loss' not in key and 'val_' in key]
    for metric in metrics:
        plt.subplot(1, 2, 2)
        plt.plot(history.history[metric.replace('val_', '')], label=f'Train {metric}')
        plt.plot(history.history[metric], label=f'Validation {metric}')
        plt.title('Model Metrics')
        plt.ylabel('Metric Value')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("logs/" + str(date_time) + "/metrics.png")
    plt.show()
    
    return plt
    
def calculate_metrics(y_true, y_pred):
    # Calculate the accuracy metrics
    acc_125 = threshold_accuracy(y_true, y_pred, 1.25).numpy()
    acc_15625 = threshold_accuracy(y_true, y_pred, 1.25**2).numpy()
    acc_1953125 = threshold_accuracy(y_true, y_pred, 1.25**3).numpy()
    rmse = root_mean_squared_error(y_true, y_pred).numpy()
    are = absolute_relative_error(y_true, y_pred).numpy()

    return acc_125, acc_15625, acc_1953125, rmse, are 

def threshold_accuracy(y_true, y_pred, threshold=1.25):

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    depth_true = y_true[:,:,:,0]
    depth_pred = y_pred[:,:,:,0]

    ratio = tf.maximum(depth_true / depth_pred, depth_pred / depth_true)
    return tf.reduce_mean(tf.cast(ratio < threshold, tf.float32))

def root_mean_squared_error(y_true, y_pred): # RMSE on depth
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    depth_true = y_true[:,:,:,0]
    depth_pred = y_pred[:,:,:,0]

    # Calculate Root Mean Squared Error
    rmse = tf.sqrt(tf.reduce_mean(tf.square(depth_true - depth_pred)))
    return rmse

def absolute_relative_error(y_true, y_pred): # ARE on depth
    # Avoid division by zero
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    depth_true = y_true[:,:,:,0]
    depth_pred = y_pred[:,:,:,0]

    epsilon = tf.keras.backend.epsilon()  # Small constant for numerical stability
    # Calculate Absolute Relative Error
    are = tf.abs(depth_true - depth_pred) / (depth_true + epsilon)
    return tf.reduce_mean(are)

def plot_inference(infer_refracted, infer_reference, infer_true_output, infer_predictions, date_time):
    fig, axes = plt.subplots(3, 4, figsize=(12, 12))
    for i in range(3):
        true_output_expanded = tf.expand_dims(infer_true_output[i], axis=0)
        predictions_expanded = tf.expand_dims(infer_predictions[i], axis=0)
        acc_metrics = calculate_metrics(true_output_expanded, predictions_expanded)
        acc_125, acc_15625, acc_1953125, rmse, are = acc_metrics
        metric_str = f"RMSE: {rmse:.3f} \n ARE: {are:.3f}"
        
        axes[i, 0].imshow(infer_refracted[i].numpy())
        axes[i, 0].set_title('Refracted Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(infer_reference[i].numpy())
        axes[i, 1].set_title('Reference Image')
        axes[i, 1].axis('off')

        im_pred = axes[i, 2].imshow(infer_predictions[i, :, :, 0], cmap='viridis')
        axes[i, 2].set_title('Predicted Depth')
        axes[i, 2].axis('off')

        im_gt = axes[i, 3].imshow(infer_true_output[i, :, :, 0], cmap='viridis')
        axes[i, 3].set_title('Ground Truth Depth')
        axes[i, 3].axis('off')

        plt.colorbar(im_pred, ax=axes[i, 2])
        plt.colorbar(im_gt, ax=axes[i, 3])
        
        axes[i, 2].annotate(metric_str, xy=(0.5, -0.1), xycoords='axes fraction', ha='center', 
                                va='top', fontsize=14, color='black', backgroundcolor='white')

    plt.tight_layout()
    plt.savefig("logs/" + str(date_time) + "/inference.png")
    plt.show()
    return plt

def save_history(history, filename):
    with open(filename, 'w') as f:
        # Header
        f.write('Epoch, Training Loss, Root Mean Squared Error, Absolute Relative Error, Accuracy 125, Accuracy 15625, Accuracy 1953125, Validation Loss, Validation Root Mean Squared Error, Validation Absolute Relative Error, Validation Accuracy 125, Validation Accuracy 15625, Validation Accuracy 1953125\n')
        
        # Data
        for i in range(len(history.history['loss'])):
            f.write(f"{i+1}, "
                    f"{history.history['loss'][i]}, "
                    f"{history.history['root_mean_squared_error'][i]}, "
                    f"{history.history['absolute_relative_error'][i]}, "
                    f"{history.history['accuracy_125'][i]}, "
                    f"{history.history['accuracy_15625'][i]}, "
                    f"{history.history['accuracy_1953125'][i]}, "
                    f"{history.history['val_loss'][i]}, "
                    f"{history.history['val_root_mean_squared_error'][i]}, "
                    f"{history.history['val_absolute_relative_error'][i]}, "
                    f"{history.history['val_accuracy_125'][i]}, "
                    f"{history.history['val_accuracy_15625'][i]}, "
                    f"{history.history['val_accuracy_1953125'][i]}\n")

    
def exp_f(x,a,b,c):
    return a*np.exp(-(x/b)) + c;
 
def pol2_f(x,a,b,c):
    return a*x + b*x**2 + c;
 
def Puff_profile(x, y, ta, depth = -0.00326456, dx=0, dy=0, width=1):
    # evaluation of surface deformation extracted from EXP_ID=142
    upper = exp_f(ta,-6.17761515e-05,  1.78422769e+00,  7.40139158e-05)
    lower = exp_f(ta,0.00377156,  1.45234773, depth)
    a     = pol2_f(ta,0.31294945, -0.00963803,  2.6743828)
    b     = pol2_f(ta,38.56906702,  -1.6278976 , 453.87937763)
    r = np.sqrt((x-dx)**2 + (y-dy)**2)/width
    
    # scaled logistic function describing surface deformation
    return lower + (upper - lower) / (1 + np.exp(a - b * r))

def grad_puff_profile(x, y, ta, depth = -0.00326456, dx=0, dy=0, width=1):
    upper = exp_f(ta,-6.17761515e-05,  1.78422769e+00,  7.40139158e-05)
    lower = exp_f(ta,0.00377156,  1.45234773, depth)
    a     = pol2_f(ta,0.31294945, -0.00963803,  2.6743828)
    b     = pol2_f(ta,38.56906702,  -1.6278976 , 453.87937763)
    
    r = np.sqrt((x-dx)**2 + (y-dy)**2)/width
    d = b * np.exp(a - b * r)*(upper - lower) / (1 + np.exp(a - b * r))/r
    dx = d * x
    dy = d * y
    return (dx, dy)

def plot_loss(history_file):
    # Load data from a text file
    data = pd.read_csv(history_file, header=0)
    # Remove any spaces from column names
    data.columns = data.columns.str.strip()

    # Extract data for plotting
    epochs = data['Epoch']
    training_loss = data['Training Loss']
    validation_loss = data['Validation Loss']

    # Create a plot with a logarithmic y-axis
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Show the plot
    plt.savefig("logs/loss.png")
    plt.show()


if __name__ == "__main__":
    plot_loss("/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/t-rex/cnn/logs/20240603-181315/training_history.txt")
