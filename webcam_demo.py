import cv2
import numpy as np
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector


cap = cv2.VideoCapture(0) #创建一个 VideoCapture 对象 
pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=True)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

while(cap.isOpened()):#循环读取每一帧
    bgr_image = cap.read()[1]
    bgr_image = cv2.flip(bgr_image,1,dst=None)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    bboxs, landmarks = mtcnn_detector.detect_face(bgr_image)

    for cords in bboxs:
        x1,y1,x2,y2,s = cords
    
        bgr_image = cv2.rectangle(bgr_image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0))        

    cv2.imshow('detection result', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release() #释放摄像头
cv2.destroyAllWindows()#删除建立的全部窗口
