import os
import sys
import numpy as np
import pickle

DATA_PATH = "/scratch/thesis/data/scenes/apartments.pickle"


class Block():
    def __init__(self, num_classes, npoints=8192, split='train'):
        self.npoints = npoints
        self.num_classes = num_classes
        self.split = split

        with open(DATA_PATH, "rb") as f:
            self.allpoints, self.alllabels = pickle.load(f)

        if split=='train':
            labelweights = np.zeros(self.num_classes)
            for seg in self.alllabels:
                tmp,_ = np.histogram(seg,range(self.num_classes+1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(self.num_classes)

    def __getitem__(self, index):
        point_set = self.allpoints[index]
        semantic_seg = np.squeeze(self.alllabels[index])
        coordmax = np.max(point_set[:,0:3], axis=0)
        coordmin = np.min(point_set[:,0:3], axis=0)
        samplemin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
        samplemin[2] = coordmin[2]
        samplesize = np.minimum(coordmax-samplemin,[1.5,1.5,3.0])
        samplesize[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            currcenter = point_set[np.random.choice(len(semantic_seg),1)[0],0:3]
            currmin = currcenter-[0.75,0.75,1.5]
            currmax = currcenter+[0.75,0.75,1.5]
            currmin[2] = coordmin[2]
            currmax[2] = coordmax[2]
            currchoice = np.sum((point_set[:,0:3] >=(currmin-0.2))*(point_set[:,0:3] <=(currmax+0.2)),axis=1)==3
            curr_point_set = point_set[currchoice,:]
            curr_semantic_seg = semantic_seg[currchoice]
            if len(curr_semantic_seg)==0:
                continue
            mask = np.sum((curr_point_set[:,0:3] >= (currmin-0.01))*(curr_point_set[:,0:3] <= (currmax+0.01)),axis=1)==3
            vidx = np.ceil((curr_point_set[mask,0:3]-currmin)/(currmax-currmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(curr_semantic_seg>0)/len(curr_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            if isvalid:
                break
        choice = np.random.choice(len(curr_semantic_seg), self.npoints, replace=True)
        point_set = curr_point_set[choice,:]
        semantic_seg = curr_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask

        # fout = open(os.path.join(BASE_DIR, 'block.obj'), 'w')
        # for i in range(point_set.shape[0]):
        #     color = label_util.label2color(semantic_seg[i])
        #     fout.write('v %f %f %f %d %d %d\n' % (point_set[i,0],point_set[i,1], point_set[i,2], color[0], color[1], color[2]))
        # fout.close()

        return point_set, semantic_seg, sample_weight
    
    def __len__(self):
        return len(self.allpoints)


class WholeScene():
    def __init__(self, num_classes, npoints=8192, split='train'):
        self.npoints = npoints
        self.num_classes = num_classes
        self.split = split

        with open(DATA_PATH, "rb") as f:
            self.allpoints, self.alllabels = pickle.load(f)

        if split=='train':
            labelweights = np.zeros(self.num_classes)
            for seg in self.alllabels:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(self.num_classes-1)
    def __getitem__(self, index):
        point_set_ini = self.allpoints[index]
        semantic_seg_ini = np.squeeze(self.alllabels[index].astype(np.int32))
        coordmax = np.max(point_set_ini[:,0:3],axis=0)
        coordmin = np.min(point_set_ini[:,0:3],axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*1.5,j*1.5,0]
                curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini[:,0:3]>=(curmin-0.2))*(point_set_ini[:,0:3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set[:,0:3]>=(curmin-0.001))*(cur_point_set[:,0:3]<=(curmax+0.001)),axis=1)==3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask)/float(len(mask))<0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.allpoints)
