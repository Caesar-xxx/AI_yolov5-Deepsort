# -*- coding:utf-8 -*-
"""
作者：知行合一
日期：2019年 09月 12日 14:15
文件名：study_demo_course.py
地点：changsha
"""
"""
路面车辆分析系统
"""

# 导入相关包
import cv2
import numpy as np
# 导入自定义模块
from highway_detection import Detector,Tracker,SpeedEstimate


class CourseDemo:
    def __init__(self):
        # 记录各个ID的上帧位置以及当前速度
        self.left_track_id_info = {} # 字典 {12：{‘last_pos’:(30.12,1201.56),'speed':95.55}}
        self.right_track_id_info = {} # 字典 {12：{‘last_pos’:(30.12,1201.56),'speed':95.55}}

        # 实例化速度估计
        self.speed_esti = SpeedEstimate()


    def plot(self,img,tracker_bboxes,side='left'):
        """
        绘制画面
        """
        track_id_info = self.left_track_id_info if side == 'left' else self.right_track_id_info
        for (l, t, r, b, cls_name, track_id) in tracker_bboxes:
            # 绘制检测框
            cv2.rectangle(img,(l,t),(r,b),(0,255,0),2)

            # 获取车尾部中心点坐标
            
            car_pos = int(l+(r-l)/2),t
            
            # cv2.circle(img,car_pos,4,(0,0,255),-1)

            # 绘制文字
            # text = '{}-{}'.format(cls_name,track_id)

            # cv2.putText(img,text,(l,t-10),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),1)

            if track_id in track_id_info and track_id_info[track_id]['speed'] != 0:
                speed_text = track_id_info[track_id]['speed']
                cv2.putText(img,str(round(speed_text,2)),(l,t-10),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),1)
            else:
                text = '{}-{}'.format(cls_name, track_id)
                cv2.putText(img, text, (l, t - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    def update_frame_info(self,tracker_bboxes,side='left'):
        # 上次1，2，3，4
        # 本次     3，4，5，6
        # 刷新34位置和速度，插入56位置信息
        last_frame_info = (self.left_track_id_info if side == 'left' else self.right_track_id_info)

        # 本帧信息
        this_frame_info = {}
        # 获取本次的位置信息
        for (l, t, r, b, cls_name, track_id) in tracker_bboxes:

            # 获取车尾部中心点坐标
            if side == 'left':
                car_pos = int(l + (r - l) / 2), t
            else:
                car_pos = int(l + (r - l) / 2), b
                
            this_frame_info[track_id] = {'last_pos':car_pos,'speed':0}


        if len(last_frame_info) > 0:
            # 更新
            # 遍历本次记录
            update_frame_info = {}
            for key,value in this_frame_info.items():
                # 判断是否在上次记录
                if key in last_frame_info:
                    # 刷新
                    # 本帧位置
                    this_frame_pos = value['last_pos']
                    # 上帧位置
                    last_frame_pos = last_frame_info[key]['last_pos']
                    # 计算距离
                    distance = self.speed_esti.pixelDistance(this_frame_pos[0],this_frame_pos[1],last_frame_pos[0],last_frame_pos[1])
                    # 转为km/h
                    speed = distance * 3.6
                    update_frame_info[key] = {'last_pos':this_frame_pos,'speed':speed}

                else:
                    # 新增
                    this_frame_pos = value['last_pos']
                    update_frame_info[key] = {'last_pos': this_frame_pos, 'speed': 0}
            # 赋值
            last_frame_info = update_frame_info



        else:
            # 插入本帧记录
            last_frame_info = this_frame_info

        # 交换
        if side == 'left':
            self.left_track_id_info = last_frame_info
        else:
            self.right_track_id_info = last_frame_info

    def detect(self):
        # 读取视频流
        cap = cv2.VideoCapture('./videos/highway.mp4')

        # 获取帧数FPS
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        # 实例化检测器
        yolo_detector = Detector()
        # 实例化2个追踪器
        tracker_left = Tracker()
        tracker_right = Tracker()

        # 记录帧数
        frame_index = 0
        while True:
            ret,frame  = cap.read()

            if frame is None:
                break

            # 保证缩放与获取参照点一致
            frame = cv2.resize(frame,(560,1000))

            # 画面按照车道分为两部分，避免id_switch现象
            # cv2.line(frame,(281,0),(281,1000),(0,255,0),2)
            frame_left = frame[:,:281]
            frame_right = frame[:,281:]

            # 目标检测
            yolo_bboxes_left = yolo_detector.yolo_detect(frame_left)
            yolo_bboxes_right = yolo_detector.yolo_detect(frame_right)
            # 目标追踪
            tracker_bboxes_left = tracker_left.update_tracker(frame_left,yolo_bboxes_left)
            tracker_bboxes_right = tracker_right.update_tracker(frame_right,yolo_bboxes_right)

            # 每隔一秒钟计算速度
            if frame_index % fps == 0:
                # 刷新信息
                self.update_frame_info(tracker_bboxes_left,'left')
                self.update_frame_info(tracker_bboxes_right,'right')


            # 绘制结果
            self.plot(frame_left,tracker_bboxes_left,'left')
            self.plot(frame_right,tracker_bboxes_right,'right')

            frame[:, :281] = frame_left
            frame[:, 281:] = frame_right

            # 显示出来
            cv2.imshow('video',frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()





course = CourseDemo()
course.detect()