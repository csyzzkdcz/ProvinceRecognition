import tensorflow as tf
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from PIL import Image
import os
from pylab import *
from tensorflow.python.framework import ops
ops.reset_default_graph()

def rotate(list1, num):
    b = [[0 for i in range(num)] for j in range(num)]
    for i in range(num):
        for j in range(num):
            b[num-1-j][num-1-i] = list1[i][num-1-j]
    for i in range(num):
        for j in range(num):
            list1[i][j]=b[i][j]            
    return list1
#load a color image 
def Convert_Image(im):
#convert to grey level image 
    Lim=im.convert('L')
#setup a converting table with constant threshold 
    threshold=80 
    table=[]
    for i in range( 256 ):
        if i<threshold:
            table.append(0)
        else :
            table.append( 1 )
 #  convert to binary image by the table 
    bim=Lim.point(table,'1')
    width = bim.size[0]
    height = bim.size[1]
#    print(width)
#    print(height)
    flag=0
    for h in range(height):
        if flag==0:
            for w in range(width):  
                pixel = bim.getpixel((w, h))
                if pixel==0:
                    top=h
                    flag=1
    flag=0
    for h in range(height):
        if flag==0:
            for w in range(width):  
                pixel = bim.getpixel((w, height-h-1))
                if pixel==0:
                    down=height-h-1
                    flag=1
    flag=0
    for w in range(width):
        if flag==0:
            for h in range(height):
                pixel = bim.getpixel((w, h))
                if pixel==0:
                    left=w
                    flag=1
    flag=0
    for w in range(width):
       if flag==0:
           for h in range(height):
               pixel = bim.getpixel((width-w-1, h))
               if pixel==0:
                  right=width-w-1
                  flag=1
    bim2=bim.crop((left,top,right,down))
    bim3=bim2.convert('L')
    #bim3.show()
    bim4=bim3.resize((28,28),Image.ANTIALIAS)
    mtr=np.transpose(np.array(bim4,dtype=np.float))
    mtr1=mtr.reshape((1,28*28))
    for i in range(28*28):
        mtr1[0][i]=255-mtr1[0][i]
        mtr1[0][i]=mtr1[0][i]/255
    return mtr1


'''计算准确度函数'''
def compute_accuracy(xs,ys,X,y,keep_prob,sess,prediction,on):
    '''param on:on=1 means plot images'''
    y_pre = sess.run(prediction,feed_dict={xs:X,keep_prob:1.0})       # 预测，这里的keep_prob是dropout时用的，防止过拟合
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))  #tf.argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值,即为对应的数字，tf.equal 来检测我们的预测是否真实标签匹配
    if on==1:
            n1=[]
            n2=[]
            for j in range(len(y)):
                c1=np.where(y_pre[j]==np.max(y_pre[j]))
                c2=np.where(y[j]==np.max(y[j]))
                n1.append(c1[0][0])
                n2.append(c2[0][0])
            plot_images(X,n1,n2)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 平均值即为准确度
    result = sess.run(accuracy,feed_dict={xs:X,ys:y,keep_prob:1.0})
    return result  
'''权重初始化函数'''
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)  # 使用truncated_normal进行初始化
    return tf.Variable(inital)

'''偏置初始化函数'''
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)  # 偏置定义为常量
    return tf.Variable(inital)

'''卷积函数'''
def conv2d(x,W):#x是图片的所有参数，W是此卷积层的权重
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动1步，y方向运动1步

'''池化函数'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME')#池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]
'''运行主函数'''

def test(X):
#    mydata = spio.loadmat('mydata.mat')
#    X = mydata['B']
#    y = mydata['Y']
#    
#    mydata2 = spio.loadmat('mydata2.mat')
#    X2 = mydata2['B']
#    y2 = mydata2['Y']

    xs = tf.placeholder(tf.float32,[None,28*28])  # 输入图片的大小，28x28=784
    ys = tf.placeholder(tf.float32,[None,32])   # 输出0-9共10个数字
    keep_prob = tf.placeholder(tf.float32)      # 用于接收dropout操作的值，dropout为了防止过拟合
    x_image = tf.reshape(xs,[-1,28,28,1])       #-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3
        
    '''第一层卷积，池化'''
    W_conv1 = weight_variable([5,5,1,32])  # 卷积核定义为5x5,1是输入的通道数目，32是输出的通道数目
    b_conv1 = bias_variable([32])          # 每个输出通道对应一个偏置
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # 卷积运算，并使用ReLu激活函数激活
    h_pool1 = max_pool_2x2(h_conv1)        # pooling操作 
    '''第二层卷积，池化'''
    W_conv2 = weight_variable([5,5,32,64]) # 卷积核还是5x5,32个输入通道，64个输出通道
    b_conv2 = bias_variable([64])          # 与输出通道一致
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    '''全连接层'''
    h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])   # 将最后操作的数据展开
    W_fc1 = weight_variable([7*7*64,1024])            # 下面就是定义一般神经网络的操作了，继续扩大为1024
    b_fc1 = bias_variable([1024])                     # 对应的偏置
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)  # 运算、激活（这里不是卷积运算了，就是对应相乘）
    '''dropout'''
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)       # dropout操作
    '''最后一层全连接'''
    W_fc2 = weight_variable([1024,32])                # 最后一层权重初始化
    b_fc2 = bias_variable([32])                       # 对应偏置

    v1 = tf.Variable(1, name="v1")
    v2 = tf.Variable(1, name="v2")        
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)  # 使用softmax分类器
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))  # 交叉熵损失函数来定义cost function
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 调用梯度下降 
    
    saver = tf.train.Saver()

    with tf.Session() as sess:      
        homedir = os.getcwd()
        saver.restore(sess, homedir+"/supermodel.tfmodel") #会将已经保存的变量值resotre到 变量中。              
        print ("Model restored.")
#        print("v1:", sess.run(v1)) # 打印v1、v2的值和之前的进行对比
#        print("v2:", sess.run(v2))
        y_pre = sess.run(prediction,feed_dict={xs:X,keep_prob:1.0})
        c1=np.where(y_pre[0]==np.max(y_pre[0]))
#        gray()
#        imshow(X.reshape((28,28)))
#        show()
#        print(y_pre[0][17])
#        print(y_pre[0][3])
#        print(y_pre[0][8])
        print(y_pre[0][c1[0][0]])
        print(c1[0][0])
        if c1[0][0]==0:
            print("北京")
        if c1[0][0]==1:
            print("上海")
        if c1[0][0]==2:
            print("福建")
        if c1[0][0]==3:
            print("广东")
        if c1[0][0]==4:
            print("天津")
        if c1[0][0]==5:
            print("重庆")
        if c1[0][0]==6:
            print("海南")
        if c1[0][0]==7:
            print("四川") 
        if c1[0][0]==8:
            print("湖南") 
        if c1[0][0]==9:
            print("辽宁") 
        if c1[0][0]==10:
            print("云南") 
        if c1[0][0]==11:
            print("贵州")  
        if c1[0][0]==12:
            print("吉林") 
        if c1[0][0]==13:
            print("河北") 
        if c1[0][0]==14:
            print("青海")  
        if c1[0][0]==15:
            print("甘肃") 
        if c1[0][0]==16:
            print("河南")
        if c1[0][0]==17:
            print("湖北")
        if c1[0][0]==18:
            print("江西")
        if c1[0][0]==19:
            print("台湾")
        if c1[0][0]==20:
            print("黑龙江")
        if c1[0][0]==21:
            print("山东")
        if c1[0][0]==22:
            print("内蒙古")
        if c1[0][0]==23:
            print("宁夏")
        if c1[0][0]==24:
            print("山西")
        if c1[0][0]==25:
            print("陕西") 
        if c1[0][0]==26:
            print("新疆") 
        if c1[0][0]==27:
            print("西藏")
        if c1[0][0]==28:
            print("安徽")
        if c1[0][0]==29:
            print("浙江") 
        if c1[0][0]==30:
            print("广西") 
        if c1[0][0]==31:
            print("江苏") 
mydata2 = spio.loadmat('mydata2.mat')

im=Image.open('1.jpg ')
imshow(im)
show()
#plt.imshow(im)
#plt.show()
X2=Convert_Image(im)
#X2 = mydata2['B']
test(X2[0:1])