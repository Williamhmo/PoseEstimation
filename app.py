import streamlit as st
import mediapipe as mp
import numpy as np
import pickle
import cv2
import tensorflow as tf
from matplotlib  import pyplot as plt
from PIL import Image
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose
pose=mp_pose.Pose()
import tempfile
from streamlit_webrtc import WebRtcMode,webrtc_streamer
import threading
import webbrowser

import pickle
SVCmodel=pickle.load(open('SVCmodel_76.5.pkl','rb'))
def SVC(a):
    y = SVCmodel.predict([a])
    d=y[0]
    return d

import keras
model=keras.models.load_model("modelCNN_7823.h5")
def CNN(a):
    y = model.predict([a])
    num = int(np.argmax(y, axis=1))
    return num

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame

def realtime():
    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
    Action=['Walking','Running','Standing','Kicking']
    imageLocation=st.empty()
    fig, ax = plt.subplots(1, 1)

    while ctx.state.playing:
        with lock:
            img = img_container["img"]
        if img is None:
            continue

        img_array=np.array(img)
        p=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
        results=pose.process(p)
        temp=[]
        try:
            landmarks = results.pose_landmarks.landmark
        except AttributeError:
            pass
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(p,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,124,66),thickness=2,circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2)
                                    )

        for j in landmarks:
            temp = temp + [j.x, j.y, j.z, j.visibility]
        y = model.predict([temp])
        num = int(np.argmax(y, axis=1))
        # d=y[0]

        if landmarks[0] and landmarks[25] and landmarks[26] in landmarks:
            cv2.putText(p,Action[num],(10,40), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        
        imageLocation.image(p)
        
def uploadvideo():
    video =st.file_uploader('Upload video.')
    imageLocation=st.empty()
    image_array=[]
    if video is not None:
        tfile=tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())
        cap=cv2.VideoCapture(tfile.name)
        Action=['Walking','Running','Standing','Kicking']
        try:
            with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
                c=1
                while True:
                    ret,frame=cap.read()
                    if c%5==0:
                        frame0 = cv2.flip(frame, 1)
                        frame1=np.array(frame0)
                        image=cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
                        results=pose.process(image)
                        landmarks=results.pose_landmarks.landmark
                        mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245,124,66),thickness=2,circle_radius=2),
                                                mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2))

                        temp=[]
                        for j in landmarks:
                            temp = temp + [j.x, j.y, j.z, j.visibility]

                        y = model.predict([temp])
                        num = int(np.argmax(y, axis=1))
                        # d=y[0]
                        # print(d)

                        if landmarks[0] in landmarks:
                            image_height,image_width,_=image.shape
                            x=results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x*image_width
                            y=results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y*image_height   
                            cv2.putText(image,Action[num],(int(x)-50,int(y)-50), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                        # output=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#                         st.image(image)
                        st.write()
#                         imageLocation.image(image)
#                         image_array.append(image)
#                         st.write(len(image_array))

                    c+=1
            for i in len(image_array):
                imageLocation.image(image_array[i])
        except:
            st.write('Success!')
    for i in len(image_array):
        imageLocation.image(image_array[i])
                
                
                
                
def uploadimage():
    Action=['Walking','Running','Standing','Kicking']
    image=st.file_uploader('Upload an image.')
    if image is not None:
        photo=Image.open(image)
        img_array=np.array(photo)
        st.image(photo)
        st.success('Your photo is successfully uploaded.')
        p=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
        results=pose.process(p)
        temp=[]
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            mp_drawing.draw_landmarks(p,results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,124,66),thickness=2,circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(0,255,0),thickness=2,circle_radius=2)
                                     )

        for j in landmarks:
            temp = temp + [j.x, j.y, j.z, j.visibility]
#         y = model.predict([temp])
#         num = int(np.argmax(y, axis=1))
        Model_Option= st.sidebar.selectbox('Model_Options',('Convolutional Neural Network','Support Vector Classifier'))
        if Model_Option == 'Convolutional Neural Network':
            num=CNN(temp)
            
        if Model_Option == 'Support Vector Classifier':
            num=CNN(temp)
        # d=y[0]
        # st.write(type(num))
        # if landmarks[0] in landmarks:
        #     image_height,image_width=photo.size
        #     x1=results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x*image_height
        #     y1=results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y*image_width

        cv2.putText(p,Action[num],(20,40), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
        output=cv2.cvtColor(p,cv2.COLOR_BGR2RGB)
        st.image(output)
        st.write(Action[num])
        
def realtimemovenet():
    st.write('This model is only working in jupyter notebook!')
    st.write('Our team will try as soon as possible to use in this application...')
    new=1
    url=''
    if st.button('Jupyter Notebook'):
        webbrowser.open(url,new=new)
        
def main():
    menu=['Home','Contact developer']
    # sidebarImg=Image.open('pic1.jpg')
#     st.sidebar.image(sidebarImg)
    choice=st.sidebar.selectbox('Menu',menu)
    
    if choice == 'Home':
        st.header('Human Action Recognition')
        Upload_Option= st.sidebar.selectbox('Upload Options',('Upload Image','Upload Video','Real_time_Detection(Sigle Person Pose Estimation)','Real_time_Detection(Multi-person Pose Estimation)'))
        if Upload_Option == 'Upload Image':
            uploadimage()
            
        if Upload_Option == 'Upload Video':
            uploadvideo()
            
        if Upload_Option == 'Real_time_Detection(Sigle Person Pose Estimation)':
            realtime()
            
        if Upload_Option == 'Real_time_Detection(Multi-person Pose Estimation)':
            realtimemovenet()
                                 
    if choice == 'Contact developer':
        st.write('Thanks for using our application...')
if __name__ =='__main__':
    main()
