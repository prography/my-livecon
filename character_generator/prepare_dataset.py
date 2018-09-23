import cv2
import os

def video_to_frames(dataroot, movie_name):
    videofiles = os.listdir(dataroot)
    videofiles = [os.path.join(dataroot, v) for v in videofiles]

    os.makedirs('frames', exist_ok=True)

    for idx in range(len(videofiles)):
        cap = cv2.VideoCapture(videofiles[idx])

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if ret:
                print("frame num:", frame_num)
                if frame_num % 15 == 0: # 똑같은 프레임 너무 많이 뽑혀서 15 프레임 당 하나만 저장하게 해놨는데 알아서 바꾸면 됨!
                    cv2.imwrite('frames\\%s_%d_%d.png' % (movie_name, movie_name, 1, frame_num), frame)
                frame_num += 1
            else:
                break

        print("Video %d frame converting completed!" % idx)

if __name__ == "__main__":
    movie_name = 'kiki'
    os.makedirs(movie_name, exist_ok=True)
    video_dir = os.path.join("D:\\Deep_learning\\Data\\멘토_LiveCon\\videos", movie_name)
    video_to_frames(video_dir, movie_name)