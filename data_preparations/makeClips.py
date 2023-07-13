import os
import os.path as osp
import shutil
import random
import numpy as np
from multiprocessing import Pool    #多进程库

test_base = r"/home/um202070049/share/movieNet/orderedData"
#test_base = r"D:\data\samples\orderedData"
def makeClipsByMovie(movieName):
    global test_base
    #_, movieName = osp.split(moviePath)
    moviePath = osp.join(test_base, movieName, 'all')
    framesName = os.listdir(moviePath)
    framesName.sort(key=lambda x: x[:-4])
    #print(framesName)
    frameNums = len(framesName)
    train_clips = osp.join(test_base, movieName, 'train')
    val_clips = osp.join(test_base, movieName, 'val')
    test_clips = osp.join(test_base, movieName, 'test')
    try:
        os.mkdir(train_clips)
        os.mkdir(test_clips)
        os.mkdir(val_clips)
        for i in range(10):
            L, R = i * frameNums // 10, (i+1) * frameNums // 10
            startFrames = np.random.randint(L, R, 5)
            for j in range(5):
                clip = framesName[startFrames[j]: startFrames[j]+frameNums // 10]
                if j < 3: clipPath = osp.join(train_clips, 'clip_'+str(i)+str(j))
                elif j == 3: clipPath = osp.join(val_clips, 'clip_'+str(i)+str(j))
                else: clipPath = osp.join(test_clips, 'clip_'+str(i)+str(j))
                os.mkdir(clipPath)
                for fn in clip:
                    src = osp.join(moviePath, fn)
                    shutil.copy(src, clipPath)  #将src文件拷贝到clipPath文件夹下
    except OSError: pass
if __name__ == '__main__':
    #movies = osp.join(test_base, "test_movies")
    #moviePaths = [osp.join(movies, _) for _ in os.listdir(movies)]
    movieNames = os.listdir(test_base)
    with Pool(50) as p:
        p.map(makeClipsByMovie, movieNames)
