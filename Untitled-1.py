# %%
import numpy as np
import nibabel as nib
import glob
import keras

from keras.utils import to_categorical
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# %%
DATASET_PATH = 'E:/NeuralNetworks/datasets/BraTS2020/'
TRAIN_DATASET_PATH = DATASET_PATH + 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALID_DATASET_PATH = DATASET_PATH + 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/'

# %%
image_flair = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
print(image_flair.shape)
+
image_t1 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
image_t1ce = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
image_t2 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
image_mask = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata().astype(np.uint8)

image_flair = scaler.fit_transform(image_flair.reshape(-1, image_flair.shape[-1])).reshape(image_flair.shape)
image_t1 = scaler.fit_transform(image_t1.reshape(-1, image_t1.shape[-1])).reshape(image_t1.shape)
image_t1ce = scaler.fit_transform(image_t1ce.reshape(-1, image_t1ce.shape[-1])).reshape(image_t1ce.shape)
image_t2 = scaler.fit_transform(image_t2.reshape(-1, image_t2.shape[-1])).reshape(image_t2.shape)

import random
n = random.randint(45,image_mask.shape[2]-45)
print(n)

plt.figure(figsize=(25,18))

plt.subplot(151)
plt.imshow(image_flair[:,:,n], cmap='gray')
plt.title('Image flair')
plt.subplot(152)
plt.imshow(image_t1[:,:,n], cmap='gray')
plt.title('Image t1')
plt.subplot(153)
plt.imshow(image_t1ce[:,:,n], cmap='gray')
plt.title('Image t1ce')
plt.subplot(154)
plt.imshow(image_t2[:,:,n], cmap='gray')
plt.title('Image t2')
plt.subplot(155)
plt.imshow(image_mask[:,:,n])
plt.title('Image mask')




# %%
print(np.unique(image_mask))
image_mask[image_mask==4] = 3
print(np.unique(image_mask))

# %%
plt.figure(figsize=(25,18))

plt.subplot(151)
plt.imshow(image_flair[:,:,n], cmap='gray')
plt.title('Image flair')
plt.subplot(152)
plt.imshow(image_t1[:,:,n], cmap='gray')
plt.title('Image t1')
plt.subplot(153)
plt.imshow(image_t1ce[:,:,n], cmap='gray')
plt.title('Image t1ce')
plt.subplot(154)
plt.imshow(image_t2[:,:,n], cmap='gray')
plt.title('Image t2')
plt.subplot(155)
plt.imshow(image_mask[:,:,n])
plt.title('Image mask')

# %%
combined_image = np.stack([image_flair, image_t1ce, image_t2],axis=3)
combined_image = combined_image[56:184, 56:184, 13:141]
image_mask = image_mask[56:184, 56:184, 13:141]

# %%
plt.figure(figsize=(25, 8))

plt.subplot(141)
plt.imshow(combined_image[:,:,n, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(142)
plt.imshow(combined_image[:,:,n, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(143)
plt.imshow(combined_image[:,:,n, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(144)
plt.imshow(image_mask[:,:,n])
plt.title('Mask')
plt.show()

# %%
np.save('sample_image.npy',combined_image)

# %% [markdown]
# ### pre-processing data

# %%
#training data pre-processing
t2_list = sorted(glob.glob(f"{TRAIN_DATASET_PATH}*/*t2.nii"))
t1ce_list = sorted(glob.glob(f"{TRAIN_DATASET_PATH}*/*t1ce.nii"))
flair_list = sorted(glob.glob(f"{TRAIN_DATASET_PATH}*/*flair.nii"))
mask_list = sorted(glob.glob(f"{TRAIN_DATASET_PATH}*/*seg.nii"))

for img in range(len(t2_list)):
    print("preparing image and mask number: ", img)
    
    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3 
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("Save Me")
        temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save(DATASET_PATH+'BraTS2020_TrainingData/input_data_3channels/images/image_'+str(img)+'.npy', temp_combined_images)
        np.save(DATASET_PATH+'BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(img)+'.npy', temp_mask)
        
    else:
        print("I am useless")  
    

# %%
import splitfolders

input_folder = DATASET_PATH+'BraTS2020_TrainingData/input_data_3channels/'
output_folder = DATASET_PATH+'input_data_128/'

splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25), group_prefix=None)


# %% [markdown]
# ### Custom Data Generator

# %%
import os

# %%
def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.array(images)
    
    return(images)

# %%
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    #keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            print(f"Loading batch from {batch_start} to {limit}")

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            print(f"Shapes - X: {X.shape}, Y: {Y.shape}")

            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

# %%
train_img_dir = DATASET_PATH + "input_data_128/train/images/"
train_mask_dir = DATASET_PATH + "input_data_128/train/masks/"
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

# %%
train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

# %%
img, msk = train_img_datagen.__next__()

# %%
img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

# n_slice=random.randint(0, test_mask.shape[2])
n_slice=60
plt.figure(figsize=(20, 8))

plt.subplot(141)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(142)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(143)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(144)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# %% [markdown]
# ### Defining the Model

# %%
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU
import random

kernel_initializer =  'he_uniform'

# %%
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
     
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
     
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
     
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model

# %% [markdown]
# ### Training the model

# %%
train_img_dir = DATASET_PATH + "input_data_128/train/images/"
train_mask_dir = DATASET_PATH + "input_data_128/train/masks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0,num_images-1)
test_img = np.load(train_img_dir+img_list[img_num])
test_mask = np.load(train_mask_dir+msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(20, 8))

plt.subplot(141)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(142)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(143)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(144)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

# %%
train_img_dir = DATASET_PATH + "input_data_128/train/images/"
train_mask_dir = DATASET_PATH + "input_data_128/train/masks/"

val_img_dir = DATASET_PATH + "input_data_128/val/images/"
val_mask_dir = DATASET_PATH + "input_data_128/val/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

batch_size = 2

# %%
train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

# %%
log = train_img_datagen.__next__()


# %%
wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# %%
steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size
print(steps_per_epoch)
print(val_steps_per_epoch)

# %%
model = simple_unet_model(IMG_HEIGHT=128,IMG_WIDTH=128, IMG_DEPTH=128, IMG_CHANNELS=3, num_classes=4)

model.compile(optimizer=optim, loss = total_loss, metrics=metrics)

print(model.input_shape)
print(model.output_shape)


# %%
history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=2,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch,
                    )

model.save('brats_3d.hdf5')

# %%
