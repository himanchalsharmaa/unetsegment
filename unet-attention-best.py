#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'headers.ipynb')
get_ipython().run_line_magic('run', 'dataset.ipynb')
get_ipython().run_line_magic('run', 'unet_plots.ipynb')
get_ipython().run_line_magic('run', 'loss.ipynb')


# #### Load Data :

# In[2]:


# LoadData_Info_with_print()


# In[3]:


LoadData_Info_without_normal_print()


# In[4]:


X_train, X_valid, X_test, y_train, y_valid, y_test = UNet_dataset_with_valid(0.7,0.1,0.2)
X_train/=255
X_valid/=255
X_test/=255
y_train/=255
y_valid/=255
y_test/=255


# In[5]:


# X_train, X_test, y_train, y_test = UNet_dataset(0.7)


# In[6]:


data[2]


# In[7]:


print(len(X_train),len(X_valid),len(X_test),len(y_train),len(y_valid),len(y_test))


# In[8]:


# len(X_train),len(X_test)


# #### Sample Plot:

# In[11]:


x=100
plot_sample(data[0][0],data[0][1],[],x,x+10)


# In[50]:


plot_sample(data[1][0],data[1][1],[],x,x+10)
# plot_sample(data[2][0],data[2][1],[],0,3)


# ## Unet Construction:

# In[13]:


get_ipython().run_line_magic('run', 'unet_architecture_1.ipynb')


# In[14]:


input_shape = (256, 256, 3)
unet_model = unet_build(input_shape)
unet_model.summary()


# In[15]:


input_shape = (256, 256, 3)
a_unet_model = attention_unet_build(input_shape)
# a_unet_model.summary()


# In[16]:


# %%html
# <img src="https://i.ibb.co/gPzyV5P/download.png" />


# In[17]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping


# In[18]:


# unet_model.compile(optimizer='adam', loss=Dice_BCELoss,metrics = [DiceBCELoss])
# unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss=Dice_BCELoss,metrics=[DiceBCELoss])
# unet_model.compile(optimizer=Adam(learning_rate=1e-5), loss=BCE_loss,metrics=[DiceBCELoss])
# unet_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss,metrics=['accuracy'])


# In[19]:


unet_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


# In[20]:


now = time.time()
mlsec  = repr(now).split('.')[1][:3]
# f_name = time.strftime("%Y-%m-%d %H.%M.%S.{}".format(mlsec), time.localtime(now))
f_name = ""
filepath=r'C:\Users\laksa\OneDrive\Desktop\cnn\callbacks'+'\\'+f_name
filepath+='_BCE-1e-3-0_new.hdf5'
filepath


# In[21]:


checkpoint              = ModelCheckpoint(filepath=filepath, monitor='val_accuracy',verbose=1,save_best_only=True)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',patience=5,verbose=1,factor=0.5,minlr=0.0000001)
# early                   = EarlyStopping(patience=6, monitor='val_loss')

# callbacks=[checkpoint]
# callbacks = [checkpoint, learning_rate_reduction]
callbacks = [checkpoint]
history = unet_model.fit(X_train,y_train,batch_size=25,epochs=100,validation_data=(X_valid,y_valid),callbacks=callbacks)
# history = unet_model.fit(X_train,y_train,batch_size=10,epochs=15,callbacks=callbacks)


# In[22]:


plot_train_val_loss_(history)
plot_train_val_acc_(history)


# In[23]:


model_best = load_model(filepath,custom_objects={'DiceBCELoss':DiceBCELoss,'BCE_loss':BCE_loss,'dice_coef':dice_coef})


# In[51]:


model_best = unet_model


# In[52]:


pred[0]
plt.imshow(pred[5])


# In[53]:


get_ipython().run_line_magic('run', 'unet_plots.ipynb')


# In[54]:


def plot_sample(X, y,z,start,val):
    fig = plt.figure(figsize=(17, 17))
    if len(z)==0 :
        index=start
        diff = val*2-start*2
        for i in range(0,diff,2):
            fig.add_subplot(val-start,2,i+1)
            plt.imshow(X[index])
            plt.axis('off')

            fig.add_subplot(val-start,2,i+2)
            plt.imshow(y[index])
            plt.axis('off')
            index+=1
    else:
        index=start
        for i in range(0,val*3-start*3,3):
            fig.add_subplot(val,3,i+1)
            plt.imshow(X[index])
            plt.axis('off')

            fig.add_subplot(val,3,i+2)
            plt.imshow(y[index])
            plt.axis('off')
            
            fig.add_subplot(val,3,i+3)
            plt.imshow(z[index])
            plt.axis('off')
            
            index+=1


# In[55]:


pred=model_best.predict(X_test)


# In[56]:


plot_sample(X_test,y_test,pred,0,10)


# In[46]:


pred_train=model_best.predict(X_train)
plot_sample(X_train,y_train,pred_train,0,10)


# In[26]:


# os.listdir(r"D:\tumor\resized\malignant")
# os.listdir(r"D:\tumor\resized\benign")
# os.listdir(r"D:\tumor\resized\normal")


# ## Predict Benign Tumor :

# In[57]:


img1=plt.imread(directory+subdir[0]+'benign (208) resized.jpg')/255
imgm1=plt.imread(directory+subdir[0]+'benign (208)_mask resized_greyscale.jpg')/255
img2=plt.imread(directory+subdir[0]+'benign (226) resized.jpg')/255
imgm2=plt.imread(directory+subdir[0]+'benign (226)_mask resized_greyscale.jpg')/255
img3=plt.imread(directory+subdir[0]+'benign (21) resized.jpg')/255
imgm3=plt.imread(directory+subdir[0]+'benign (21)_mask resized_greyscale.jpg')/255

plot_sample([img1,img2,img3],[imgm1,imgm2,imgm3],model_best.predict(np.array([img1,img2,img3])),0,3)

img4=plt.imread(directory+subdir[0]+'benign (100) resized.jpg')/255
imgm4=plt.imread(directory+subdir[0]+'benign (100)_mask resized_greyscale.jpg')/255
imgm4_1=plt.imread(directory+subdir[0]+'benign (100)_mask_1 resized_greyscale.jpg')/255

plot_sample_single([img4,imgm4,imgm4_1])

img5=plt.imread(directory+subdir[0]+'benign (181) resized.jpg')/255
imgm5=plt.imread(directory+subdir[0]+'benign (181)_mask resized_greyscale.jpg')/255
imgm5_1=plt.imread(directory+subdir[0]+'benign (181)_mask_1 resized_greyscale.jpg')/255

plot_sample_single([img5,imgm5,imgm5_1])

img6=plt.imread(directory+subdir[0]+'benign (195) resized.jpg')/255
imgm6=plt.imread(directory+subdir[0]+'benign (195)_mask resized_greyscale.jpg')/255
imgm6_1=plt.imread(directory+subdir[0]+'benign (195)_mask_1 resized_greyscale.jpg')/255
imgm6_2=plt.imread(directory+subdir[0]+'benign (195)_mask_2 resized_greyscale.jpg')/255

plot_sample_single([img6,imgm6,imgm6_1,imgm6_2])


# ## Predict Malignant Tumor :

# In[58]:


img1=plt.imread(directory+subdir[1]+'malignant (141) resized.jpg')/255
imgm1=plt.imread(directory+subdir[1]+'malignant (141)_mask resized_greyscale.jpg')/255

img2=plt.imread(directory+subdir[1]+'malignant (101) resized.jpg')/255
imgm2=plt.imread(directory+subdir[1]+'malignant (101)_mask resized_greyscale.jpg')/255

img3=plt.imread(directory+subdir[1]+'malignant (111) resized.jpg')/255
imgm3=plt.imread(directory+subdir[1]+'malignant (111)_mask resized_greyscale.jpg')/255

plot_sample([img1,img2,img3],[imgm1,imgm2,imgm3],model_best.predict(np.array([img1,img2,img3])),0,3)


# ## Predict Normal Ultrasound :

# In[59]:


img1=plt.imread(directory+subdir[2]+'normal (102) resized.jpg')/255
imgm1=plt.imread(directory+subdir[2]+'normal (102)_mask resized_greyscale.jpg')/255

img2=plt.imread(directory+subdir[2]+'normal (123) resized.jpg')/255
imgm2=plt.imread(directory+subdir[2]+'normal (123)_mask resized_greyscale.jpg')/255

img3=plt.imread(directory+subdir[2]+'normal (99) resized.jpg')/255
imgm3=plt.imread(directory+subdir[2]+'normal (99)_mask resized_greyscale.jpg')/255

img4=plt.imread(directory+subdir[2]+'normal (1) resized.jpg')/255
imgm4=plt.imread(directory+subdir[2]+'normal (1)_mask resized_greyscale.jpg')/255

plot_sample([img1,img2,img3,img4],[imgm1,imgm2,imgm3,imgm4],model_best.predict(np.array([img1,img2,img3,img4])),0,4)


# In[30]:


dice_coef(pred,y_test)


# In[ ]:




