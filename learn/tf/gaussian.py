import tensorflow as tf
from PIL import Image    
import numpy as np
import math



g = tf.Graph()

with g.as_default():
    
    def getGuessValue(kerStd,posX,posY):
        return 1./(2.*math.pi*(np.power(kerStd,2)))*math.exp(-(np.power(posX,2)+np.power(posY,2))/(2.*(np.power(kerStd,2))))
    
    def getGaussKernel(kerStd):
        K11=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,-1,1),[0.,0.,0.])),np.array([0.,0.,0.,1.])))
        K12=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,0,1),[0.,0.,0.])),np.array([0.,0.,0.,1.])))        
        K13=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,1,1),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        K21=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,-1,0),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        K22=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,0,0),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        K23=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,1,0),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        K31=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,-1,-1),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        K32=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,0,-1),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        K33=np.column_stack((np.row_stack((np.eye(3)*getGuessValue(kerStd,1,-1),[0.,0.,0.])),np.array([0.,0.,0.,1.])))      
        print(K11.shape)
        kernel=tf.constant(np.array(
                [
                    [
                       K11,
                       K12,
                       K13
                    ],
                    [
                       K21,
                       K22,
                       K23                    
                    ],
                    [
                       K31,
                       K32,
                       K33                    
                    ]              
                ])
                ,dtype=tf.float32)#3*3*4*4
        return kernel

    def getImageData(fileNameList):
        imageData=[]
        for fn in fileNameList:        
            testImage = Image.open(fn)
            testImage.show() 
            imageData.append(np.array(testImage))
        return np.array(imageData,dtype=np.float32)

    imageData=getImageData(("dog.png",))
    testData=tf.constant(imageData)
    kernel=getGaussKernel(0.8)
    y=tf.cast(tf.nn.conv2d(testData,kernel,strides=[1,1,1,1],padding="SAME"), dtype=tf.int32)
    init_op = tf.global_variables_initializer()
with tf.Session(graph=g) as sess:
    print(testData.get_shape())
    # print(kernel.eval())
    # print(kernel.get_shape())
    resultData=sess.run(y)[0]
    # print(resultData.shape)
    resulImage=Image.fromarray(np.uint8(resultData))   
    resulImage.show()
    # print(y.get_shape())