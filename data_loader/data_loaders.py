import os
import os.path as osp
import random
import numpy as np
from multiprocessing import Pool    #多进程库
import torch
import torchvision
#from PIL import Image
from skimage import io
#from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
import sys
sys.path.append(r"/home/um202070049/clip2movie")
#torch.cuda.set_per_process_memory_fraction(0.8, 0)
from data_loader.transforms import CenterCrop, Normalize, OneImageCollate, Resize, CenterCrop3D

class MyDataset(Dataset):
    def __init__(self, path_txt, mode, movie_shot=1400, clip_shot=140, num_keyframe=3, \
                    transform=Compose([
                                        Resize((256, 256)),
                                        CenterCrop((224, 224)),
                                        Normalize(
                                            mean=[123.675, 116.28, 103.53],
                                            std=[58.395, 57.12, 57.375],
                                            to_rgb=False),
                                        ToTensor(),
                                        ])
                        ):
        super(MyDataset, self).__init__()
        path_file = open(path_txt, 'r')
        self.videos = []
        for line in path_file.readlines():
            self.videos.append(line.rstrip())
        self.mode = mode
        self.movie_shot = movie_shot
        self.clip_shot = clip_shot
        self.num_keyframe = num_keyframe
        self.transform = transform
    def __getitem__(self, index):
        #import cv2
        #cv2.setNumThreads(0)
        videoDir = self.videos[index]   # D:\data\samples\orderedData\tt0032138
        movieDir, trainClipDir, valClipDir = osp.join(videoDir, 'all'), \
                                                osp.join(videoDir, 'train'), \
                                                osp.join(videoDir, 'val')

        movieTensor = torch.load(osp.join(movieDir, 'video_tensor.pth'))
        movieTensor = self.sampleVideoShot(movieTensor, self.movie_shot, self.num_keyframe)
        movieTensor = CenterCrop3D((168,168))(movieTensor)
        movieTensor = movieTensor.unsqueeze(dim=0)  #shape: 1 * frameNum * C * H * W
        if self.mode == 'train':
            posClipsTensor = self.genClipsTensor(trainClipDir)
        if self.mode == 'val':
            posClipsTensor = self.genClipsTensor(valClipDir)
        if self.mode != 'test':
            negClipList = []
            negMovie = random.sample(self.videos[:index]+self.videos[index+1:], 3)     #随机取9部电影
            for _ in negMovie:
                for i in range(9):
                    if self.mode == 'train': clipPath = osp.join(_, self.mode, 'clip_'+str(i)+'0')
                    else: clipPath = osp.join(_, self.mode, 'clip_'+str(i)+'3')
                    clipTensor = torch.load(osp.join(clipPath, 'video_tensor.pth'))
                    clipTensor = self.sampleVideoShot(clipTensor, self.clip_shot, self.num_keyframe)
                    clipTensor = CenterCrop3D((168,168))(clipTensor)
                    clipTensor = clipTensor.unsqueeze(dim=0)
                    negClipList.append(clipTensor)
            negClipsTensor = torch.cat(negClipList, dim=0)
            return movieTensor, posClipsTensor, negClipsTensor
        if self.mode == 'test':
            return movieTensor, index
        #if self.training:
        #    trainClipsTensor, valClipsTensor = self.genClipTensor(trainClipDir), self.genClipTensor(valClipDir)
        #    return movieTensor, trainClipsTensor, valClipsTensor
        #else: return movieTensor
    def __len__(self):
        return len(self.videos)
    def genClipsTensor(self, clipsDir):  #clipsDir: D:\data\samples\orderedData\tt0032138\train
        clips = os.listdir(clipsDir)    #[clip_00,...]
        clipList = []
        for i in range(len(clips)):
            clipPath = osp.join(clipsDir, clips[i])
            clipTensor = torch.load(osp.join(clipPath, 'video_tensor.pth'))
            clipTensor = self.sampleVideoShot(clipTensor, self.clip_shot, self.num_keyframe)
            clipTensor = CenterCrop3D((168,168))(clipTensor)
            clipTensor = clipTensor.unsqueeze(dim=0)
            clipList.append(clipTensor)
        clipsTensor = torch.cat(clipList, dim=0)
        return clipsTensor  # shape: clipNum * frameNum * C * H * W

    def sampleVideoShot(self, ori_videoTensor, num_shot, num_keyframe):
        total_frames = ori_videoTensor.shape[0]
        total_shots = total_frames // num_keyframe
        if total_shots >= num_shot:
            start_shotID = (total_shots - num_shot) // 2
            new_videoTensor = ori_videoTensor[start_shotID*num_keyframe: (start_shotID+num_shot)*num_keyframe]
        elif total_shots < num_shot:
            new_videoTensor = ori_videoTensor.repeat(num_shot // total_shots + 1, 1, 1, 1)[:num_shot*num_keyframe]
        return new_videoTensor
class MyDataLoader(DataLoader):
    def __init__(self, path_txt, mode, batch_size, shuffle=False, pin_memory=True, num_workers=1):

        self.path_txt = path_txt
        self.dataset = MyDataset(self.path_txt, mode)
        super(MyDataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle)
        #super(MyDataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, sampler=train_sampler)

''' 
if __name__ == '__main__':
    import time
    start = time.time()
    batch_size = 2

    path_txt=r"/home/um202070049/share/movieNet/sample_video_path.txt"
    train_dataloader = MyDataLoader(path_txt=path_txt, mode='test', batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False)
    #f = open("result.txt", 'w')
    for Movies, idx in train_dataloader:
        #movies, posClips, negClips = movies.to(self.device), posClips.to(self.device), negClips.to(self.device)
        print("Shape of Movies [M, 1, T, C, H, W]: ", Movies.shape)
        print(idx)
        #print("Shape of TrainClips [M, num_clip, T, C, H, W]: ", PosClips.shape)
        #print("Shape of ValClips [M, num_clip, T, C, H, W]: ", NegClips.shape)
        #f.write(str(Movies.shape) + '\n' + str(TrainClips.shape) + '\n' + str(ValClips.shape)+'\n')
    end = time.time()
    #print("used time: ", end - start)
    #f.write(str(end-start))
'''
