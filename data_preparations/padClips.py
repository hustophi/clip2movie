import os, glob
import os.path as osp
import shutil
from multiprocessing import Pool    #多进程库

base = r"/home/um202070049/share/movieNet/orderedData"
def padClips(movieName):
    global base
    train_clips = osp.join(base, movieName, 'train')
    val_clips = osp.join(base, movieName, 'val')
    test_clips = osp.join(base, movieName, 'test')
    clipsName = os.listdir(train_clips)    #[clip_00,...]
    for c in clipsName:
        clipPath = osp.join(train_clips, c)
        imgList = os.listdir(clipPath)  #[shot_0006_img_1.jpg,...]
        imgList.sort()
        if imgList[0].split('.')[0].split('_')[-1] == '1':
            src = osp.join(clipPath, imgList[0])
            prefix = '_'.join(imgList[0].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath, prefix+'_0.jpg')
            shutil.copy(src, dst)
        elif imgList[0].split('.')[0].split('_')[-1] == '2':
            src = osp.join(clipPath, imgList[0])
            prefix = '_'.join(imgList[0].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath, prefix+'_0.jpg')
            shutil.copy(src, dst)
            dst = osp.join(clipPath, prefix+'_1.jpg')
            shutil.copy(src, dst)
        if imgList[-1].split('.')[0].split('_')[-1] == '1':
            src = osp.join(clipPath, imgList[-1])
            prefix = '_'.join(imgList[-1].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath, prefix+'_2.jpg')
            shutil.copy(src, dst)
        if imgList[-1].split('.')[0].split('_')[-1] == '0':
            src = osp.join(clipPath, imgList[-1])
            prefix = '_'.join(imgList[-1].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath, prefix+'_1.jpg')
            shutil.copy(src, dst)
            dst = osp.join(clipPath, prefix+'_2.jpg')     
            shutil.copy(src, dst)
    clipsName = os.listdir(val_clips)
    for c in clipsName:
        clipPath = osp.join(val_clips, c)
        imgList = os.listdir(clipPath)  #[shot_0006_img_1.jpg,...]
        imgList.sort()
        if imgList[0].split('.')[0].split('_')[-1] == '1':
            src = osp.join(clipPath,  imgList[0])
            prefix = '_'.join(imgList[0].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_0.jpg')
            shutil.copy(src, dst)
        elif imgList[0].split('.')[0].split('_')[-1] == '2':
            src = osp.join(clipPath,  imgList[0])
            prefix = '_'.join(imgList[0].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_0.jpg')
            shutil.copy(src, dst)
            dst = osp.join(clipPath,  prefix+'_1.jpg')
            shutil.copy(src, dst)
        if imgList[-1].split('.')[0].split('_')[-1] == '1':
            src = osp.join(clipPath,  imgList[-1])
            prefix = '_'.join(imgList[-1].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_2.jpg')
            shutil.copy(src, dst)
        if imgList[-1].split('.')[0].split('_')[-1] == '0':
            src = osp.join(clipPath,  imgList[-1])
            prefix = '_'.join(imgList[-1].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_1.jpg')
            shutil.copy(src, dst)
            dst = osp.join(clipPath,  prefix+'_2.jpg')     
            shutil.copy(src, dst)   
    clipsName = os.listdir(test_clips)
    for c in clipsName:
        clipPath = osp.join(test_clips, c)
        imgList = os.listdir(clipPath)  #[shot_0006_img_1.jpg,...]
        imgList.sort()
        if imgList[0].split('.')[0].split('_')[-1] == '1':
            src = osp.join(clipPath,  imgList[0])
            prefix = '_'.join(imgList[0].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_0.jpg')
            shutil.copy(src, dst)
        elif imgList[0].split('.')[0].split('_')[-1] == '2':
            src = osp.join(clipPath,  imgList[0])
            prefix = '_'.join(imgList[0].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_0.jpg')
            shutil.copy(src, dst)
            dst = osp.join(clipPath,  prefix+'_1.jpg')
            shutil.copy(src, dst)
        if imgList[-1].split('.')[0].split('_')[-1] == '1':
            src = osp.join(clipPath,  imgList[-1])
            prefix = '_'.join(imgList[-1].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_2.jpg')
            shutil.copy(src, dst)
        if imgList[-1].split('.')[0].split('_')[-1] == '0':
            src = osp.join(clipPath,  imgList[-1])
            prefix = '_'.join(imgList[-1].split('.')[0].split('_')[:-1])
            dst = osp.join(clipPath,  prefix+'_1.jpg')
            shutil.copy(src, dst)
            dst = osp.join(clipPath, prefix+'_2.jpg')     
            shutil.copy(src, dst)
if __name__ == '__main__':
    #movies = osp.join(test_base, "test_movies")
    #moviePaths = [osp.join(movies, _) for _ in os.listdir(movies)]
    movieNames = os.listdir(base)
    with Pool(50) as p:
        p.map(padClips, movieNames)
