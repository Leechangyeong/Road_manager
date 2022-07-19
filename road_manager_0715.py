from datetime import datetime
import socket
from tkinter import N
import cv2
from cv2 import line
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cx_Oracle

global client_socket
connect = None
cursor = None

def makeDictFactory(cursor):
    columnNames = [d[0] for d in cursor.description]
 
    def createRow(*args):
        return dict(zip(columnNames, args))
    return createRow

def insert(tableName, dict):
    if cursor is None:
        return
    sql = 'insert into' + tableName + '('
    headers = ''
    values = ''
    i = 0
    for key, value in dict.items():
        headers += key
        values += value
        if i + 1 < len(dict):
            headers += ','
            values += ','
        i += 1
    sql += headers + ')' 'values' + '(' + 'value' + ')'
    try:
        cursor.execute(sql)
        cursor.execute('commit')
        return 0
    except:
        cursor.execute('rollback')
        return 1
        
def select(tableName, wheresql = None):
    if cursor is None:
        return

    sql = "select * from " + tableName
    if wheresql is not None:
        sql += "WHERE" + wheresql
    try:
        cursor.execute(sql)
        cursor.rowfactory = makeDictFactory(cursor)
        return cursor.fetchall()
    except:
        return None
    
def init():
    global connect
    global cursor
    
    connect = cx_Oracle.connect("LANE_DETECTOR", "raontec123", "main.raontec.co.kr:22564/traf", encoding = "UTF-8")
    cursor = connect.cursor()
    
def close():
    if connect is not None:
        try:
            connect.close()
        except Exception as e:
            print(e)

class Traffic_Tracker:
    
    # 기본 데이터 세팅
    def __init__(self):
        self.Queue = []
        self.dict = {}
        self.outlier = {}
        self.bef_dict = None
        self.Allpoint = []
        
    # 원하는 순간 캡쳐 -> 그 후로 Tracing 시작
    def capture(self, cap):
        while True:
            frame = cap.read()[1]
            cv2.imshow('Traffic Check', frame)
            key = cv2.waitKey(25)
            if key == ord('c'):
                break
        cap.release()
        cv2.destroyAllWindows()
        self.frame = frame
        self.baseframe = frame.copy() # Use after clusturing

    # 받은 패킷을 모두  self.Queue에 저장 
    def addData(self):
        dataSize = int.from_bytes(client_socket.recv(4), 'big')
        jsonString = client_socket.recv(dataSize)
        self.Queue.append(jsonString)
    
    def useDB(self, data):
        self.Queue = data
    
    # Queue에 저장된 패킷을 하나씩 꺼내서 Parsing 하는 과정
    def dataParsing(self):
            data = self.Queue.pop()
            data = eval(str(data.decode()))
            time = datetime.fromtimestamp((int(data['timestamp']))/1000)
            carType, dataId, x1, x2, y1, y2 = list(data.values())[:6]
            cen_x = int((x1 + x2)/2) ; cen_y = int(((y1 + y2)/2 + y2)/2)
            self.dict[dataId] = {'type' : carType,
                                'coord' : [cen_x, cen_y], 
                                'time' : time
                                }
            self.key = dataId
  
    # 기울기 바탕으로 이상치 제거하고, 정상값으로 판별한 값들만 표시, 추가하는 기능 (제일 최근 데이터 기준)
    def car_tracing_point(self, color):
        Id = self.key
        #line_color = color[self.dict[Id]['type']]       나중에 색깔 다시 바꿔주기

        # 전에 등장했던 차들 중에서 기울기 계산 ( 전 프레임에 없었으면, 한번도 출현 X 인 경우에는 기울기 계산 불가)
        if self.outlier.get(Id):
            
            #5번 연속 Outlier는 더이상 부적절 하다고 판단
            if self.outlier[Id]['out_cnt'] < 5:
                 
                # coord1 = 직전 좌표, coord2 = 현재 좌표
                coord1 = self.bef_dict[Id]['coord']; coord2 = self.dict[Id]['coord']
                
                # 앞으로 진행헀을 경우에만
                if  coord2[1]-coord1[1] > 2:
                    tmp_grad = (coord1[0]-coord2[0])/(coord1[1]-coord2[1])
                    
                    # 이상치 측정을 위해 특정 개수는 그냥 기록
                    avg_count = 20
                    threshold_gradient = 0.8
                    if self.outlier[Id]['count'] <= avg_count:                            
                        self.outlier[Id]['gradient'] = (self.outlier[Id]['gradient'] * self.outlier[Id] ['count'] + tmp_grad) / (self.outlier[Id] ['count'] + 1)
                        self.outlier[Id]['coord_avg_x'] = (self.outlier[Id]['coord_avg_x'] * self.outlier[Id]['count'] + coord2[0]) / (self.outlier[Id]['count'] + 1)
                        self.Allpoint.append([Id] + coord2)
                        cv2.line(self.frame, coord2, coord2, (0,255,255), 2) # 구분하려고 넣어놓은거 나중에는 바꿔야함

                    else:
                        # 이상치 거르기
                        if abs(tmp_grad - self.outlier[Id]['gradient']) > threshold_gradient:
                            cv2.line(self.frame, coord2, coord2, (0, 0, 255), 2) # 구분하려고 넣어놓은거 나중에는 삭제 해야함
                            self.outlier[Id]['out_cnt'] += 1
                            return     
                        else:
                            self.outlier[Id]['gradient'] = (self.outlier[Id]['gradient'] * self.outlier[Id] ['count'] + tmp_grad) / (self.outlier[Id] ['count'] + 1)
                            self.outlier[Id]['coord_avg_x'] = (self.outlier[Id]['coord_avg_x'] * self.outlier[Id]['count'] + coord2[0]) / (self.outlier[Id]['count'] + 1)
                            self.outlier[Id]['out_cnt'] = 0
                            cv2.line(self.frame, coord2, coord2, (0,255,255), 2)
                            self.Allpoint.append([Id] + coord2) 
                    self.outlier[Id]['count'] += 1
                
        # 처음 발견된 차는 기울기 정보, 측정횟수, 이상치 연속 횟수 정보 없으므로 초기화
        else:
            self.outlier[Id] = {'gradient' : 0,
                                'coord_avg_x' : 0,
                                'count' : 0,
                                'out_cnt' : 0
                                }
    
    def get_avg_coord(self):
        tmp_list = []
        Id_list = []
        for i in list(self.dict.keys()):
            if self.outlier[i]['count'] > 5:    # 차량 좌표가 5개 이상 찍힌 Id만 추가하기
                Id_list.append(i)
                tmp_list.append([self.outlier[i]['coord_avg_x'], self.outlier[i]['gradient']])
        return tmp_list, Id_list
    
    """        
    def passed_car_delete(self):
        for Id in list(self.dict.keys()):
            check_time = self.dict[Id]['time']
            time_gap = self.time - check_time
            if time_gap.seconds >= 3:
                del self.dict[Id] 
    """
    
    def change_bef_dict(self):
        self.bef_dict = self.dict.copy()

class clusturing():
    
    def __init__(self, input_data):
        self.data = np.float32(np.array(input_data))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        cluster_num = 5
        attempts = 10
        self.ret, self.label, self.center = cv2.kmeans(self.data, cluster_num, 
                                        None,
                                        criteria,
                                        attempts,
                                        cv2.KMEANS_RANDOM_CENTERS)
           
    def clustering_point(self):
        data = self.data
        label = self.label
        A = data[label.ravel() == 0].tolist(); A = list(i[0] for i in A); A = list(map(int, A))
        B = data[label.ravel() == 1].tolist(); B = list(i[0] for i in B); B = list(map(int, B))
        C = data[label.ravel() == 2].tolist(); C = list(i[0] for i in C); C = list(map(int, C))
        D = data[label.ravel() == 3].tolist(); D = list(i[0] for i in D); D = list(map(int, D))
        E = data[label.ravel() == 4].tolist(); E = list(i[0] for i in E); E = list(map(int, E))
        self.clist = [A, B, C, D, E]
        
    def clustered_Id(self, coord_data, Id_data):
        clustered_Id = []
        coord_data = list(map(int, coord_data))
        for i in range (len(self.clist)):
            tmp = []
            for j in self.clist[i]:
                try:
                    tmp.append(Id_data[coord_data.index(j)])
                except :
                    print('e')
            clustered_Id.append(tmp)
        for i in clustered_Id:
            print(i)
            print("")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print('cId' , clustered_Id)
        self.Clu_Id = clustered_Id
    
    def show_clustered_point(self, All_point, Color_list, base_frame):
        clustered_point = []
        for i in range(len(self.Clu_Id)):
            tmp = []
            for j in All_point:
                color = Color_list[i]
                if j[0] in self.Clu_Id[i]:
                    tmp.append(j[1:])
                    cv2.line(base_frame, j[1:], j[1:], color, 2)
            clustered_point.append(tmp)
        for i in clustered_point:
            print(i)
            print("")
        self.Clu_pt = clustered_point
        return base_frame
        
class road_manager():
    def __init__(self, Clu_coord):
        self.coord = Clu_coord 
        print('Clu_coord', Clu_coord)
        
    def do_Regression(self):
        tmp_slope = []
        tmp_ycoord = []
        for i in range(len(self.coord)):
            coord = np.transpose(self.coord[i])
            x = np.array(coord[0]); y = np.array(coord[1])
            line_filter = LinearRegression()
            line_filter.fit(x.reshape(-1,1), y)
            tmp_slope.append(float(line_filter.coef_[0]))
            tmp_ycoord.append(float(line_filter.intercept_))
        self.slope = tmp_slope
        self.y_coord = tmp_ycoord
        
    def cal_line(self, frame_x, frame_y, base_frame):
        slope = self.slope
        y = self.y_coord
        print(slope, y)
        f_x = int(frame_x)
        f_y = int(frame_y)
        for i in range(len(slope)): # x = 0, x = f_x, y= 0, y = f_y
            tmp = []
            t_1 = int(y[i])
            t_2 = int(slope[i] * f_x + y[i])
            t_3 = int(-y[i] / slope[i])
            t_4 = int((f_y - y[i]) / slope[i])
            print('t', t_1, t_2, t_3, t_4)
            if 0 <= t_1 <= f_y:
                tmp.append([0, t_1])
            if 0 <= t_2 <= f_y:
                tmp.append([f_x, t_2])
            if 0 <= t_3 <= f_x:
                tmp.append([t_3, 0])
            if 0 <= t_4 <= f_x:
                tmp.append([t_4, f_y])
            print(len(tmp))
            print(tmp)
            cv2.line(base_frame, tmp[0], tmp[1], (255, 255, 255), 1)
        return base_frame
#######################################################################################################
#######################################################################################################

'''
init()
dbdata = select('VW_DET_OBJ_HIS_15M')
close()
'''

if __name__ == '__main__':
    
    # 비디오 데이터 받음
    video_path = 'https://paju.cctvstream.net/live/smart_104.stream/playlist.m3u8'
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Traffic Check', cv2.WINDOW_NORMAL)

    # 교통 정보 받음
    HOST = '10.10.20.6'
    PORT = 19999
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    
    cap = cv2.VideoCapture(video_path)
    frame_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(frame_x, frame_y)
    myroad = Traffic_Tracker()
    myroad.capture(cap)

    try:
        Color_list = [(0, 0, 255), (153, 000, 153), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        while True:
            myroad.addData()
            #myroad.useDB(dbdata)
            myroad.dataParsing()
            myroad.car_tracing_point(Color_list)
            myroad.change_bef_dict()
            cv2.imshow('video', myroad.frame)
            key = cv2.waitKey(25)
    
    except KeyboardInterrupt:
        client_socket.close()
        cap.release()
        cv2.destroyAllWindows()
    
    ###########################################################################
    # Clusturing
    
    coord_data, Id_data = myroad.get_avg_coord()
    print(coord_data)

    All_point = myroad.Allpoint
    print("All",All_point)
    print('start clusturing')
    
    cluster = clusturing(coord_data)
    cluster.clustering_point()
    coord_data = list(i[0] for i in coord_data)
    print('coord_data', coord_data)
    cluster.clustered_Id(coord_data, Id_data)
    cluster_frame = cluster.show_clustered_point(All_point, Color_list, myroad.baseframe)
    #cluster_frame = cv2.resize(cluster_frame, (0, 0), fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
    
    
    cv2.namedWindow('video_c', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow('video_c', cluster_frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    ###########################################################################
    # Drawing Road
    
    road = road_manager(cluster.Clu_pt)
    road.do_Regression()
    
    cluster_frame = road.cal_line(frame_x, frame_y, cluster_frame)
    
    cv2.namedWindow('video_c', cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow('video_c', cluster_frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()