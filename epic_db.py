from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import json, random, os.path
from glob import glob
import zipfile
import torch
import numpy as np
import pandas as pd

class EPIC_Dataset(Dataset):
    def __init__(self, db_root_dir, training=True):
        #prepare directories and annotation files
        action_annotation_file = 'actions_train_1-25_with_objectsids.csv' if training else 'actions_train_26-31_with_objectsids.csv'
        self.db_root_dir = db_root_dir
        self.frames_dir = db_root_dir + 'frames_rgb_flow/rgb/train/'
        self.annotation_dir = db_root_dir + 'annotations/'        
        self.action_table = pd.read_csv(self.annotation_dir + action_annotation_file)
        self.object_table = pd.read_pickle(self.annotation_dir + 'EPIC_train_object_labels.pkl')
        self.verb_dict = pd.read_csv(self.annotation_dir+'EPIC_verb_classes.csv')
        self.noun_dict = pd.read_csv(self.annotation_dir+'EPIC_noun_classes.csv')
        self.n_verbs, self.n_nouns = len(self.verb_dict), len(self.noun_dict)	
        self.masks_dir      = self.db_root_dir + f'masks_v2_for_AR_100x100_50/train/'      

        self.data_dir       = './input_data/'
        self.state_dict_file  = json.load(open(self.data_dir + 'state_mapping_v3.json', 'r'))
        self.state_dict       = self.state_dict_file['states_id']
        self.verb2state_dict  = self.state_dict_file['state_transitions']

        self.img_scale = (456/1920, 256/1080)
        self.gaussian_width = self.get_gaussian_filter(g_size=456, center=228)
        self.gaussian_height = self.get_gaussian_filter(g_size=256, center=128)

        self.max_sequence_of_objects = 512
        self.max_sequence_of_frames = 414
        

    def __getitem__(self, index):
        item = self.action_table.iloc[index, :] #action segment
        #objects_df =  self.object_table.iloc[eval(item.objects_uid), :]
        objects_df = self.get_objects_in_segment(item)
        #(EXP1) Unique Object list in action segment 
        #objects_list_unique = [] self.to_categorial_multilabel(np.unique(objects_df.noun_class.get_values()), len(self.noun_dict))

        #(EXP2) sorted list of object 
        #objects_in_action_segment_sorted = objects_df.sort_values('frame').noun_class.get_values()
        #objects_in_action_segment_sorted = self.pad_sequence(objects_in_action_segment_sorted)

        #(EXP 3) sorted list per frame
        #objects_per_frame = self.get_objects_per_frame(objects_df, item)

        #(EXP 4) objects per frame with time
        # objects_per_frame = self.get_objects_per_frame(objects_df, item)
        # num_frames = item.stop_frame - item.start_frame
        # time_ticks = self.get_timecounters_features(num_frames//30)
        # objects_per_frame_with_time = torch.cat([time_ticks, objects_per_frame], dim = 0)

        #(EXP 5) objects per frame with time and spatial information
        # objects_score_per_frame = self.get_objects_per_frame(objects_df, item, score_obj_position=True)
        # num_frames = item.stop_frame - item.start_frame
        # time_ticks = self.get_timecounters_features(num_frames//30)
        # objects_score_per_frame_with_time = torch.cat([time_ticks, objects_score_per_frame], dim = 0)

        #(EXP 6) objects per frame with time and spatial fovea hands.
        hands_per_frame = self.get_cocohand_position_in_segment(item)
        objects_score_per_frame = self.get_objects_per_frame(objects_df, item, score_obj_position=True, center=hands_per_frame)
        num_frames = item.stop_frame - item.start_frame
        time_ticks = self.get_timecounters_features(num_frames//30)
        objects_score_per_frame_with_time = torch.cat([time_ticks, objects_score_per_frame], dim = 0)

        #(EXP 7) objects per frame with time and state
        # objects_state_per_frame = self.get_objects_per_frame(objects_df, item, score_obj_position=False, state=True)
        #num_frames = item.stop_frame - item.start_frame
        #time_ticks = self.get_timecounters_features(num_frames//30)
        # objects_state_per_frame_with_time = torch.cat([time_ticks, objects_state_per_frame], dim = 0)

        return {
                'verb_class': item.verb_class, 
                'noun_class': item.noun_class,

                # EXP1 
                #'objects_list_unique' : objects_list_unique,
                
                # EXP2
                #'objects_list_sorted' : objects_in_action_segment_sorted

                # EXP 3 objects per frame
                # 'objects_list_sorted_perframe' : objects_per_frame.view(self.max_sequence_of_frames*self.n_nouns),
                # 'objects_list_sorted_perframe_noreshape' : objects_per_frame, #for tcn

                # EXP 4 Add temporal features 
                # 'objects_per_frame_with_time' : objects_per_frame_with_time.view(self.max_sequence_of_frames*(self.n_nouns+4)),
                # 'objects_per_frame_with_time_noreshape' : objects_per_frame_with_time, #for tcn

                #EXP 5 Add spatial information
                # 'objects_score_per_frame_with_time': objects_score_per_frame_with_time.view(self.max_sequence_of_frames*(self.n_nouns+4)),
                # 'objects_score_per_frame_with_time_noreshape': objects_score_per_frame_with_time

                #EXP 6 Add spatial information fovea hands
                'objects_score_per_frame_with_time': objects_score_per_frame_with_time.view(self.max_sequence_of_frames*(self.n_nouns+4)),
                'objects_score_per_frame_with_time_noreshape': objects_score_per_frame_with_time,

                #EXP 7 Add state and time but not spatial
                # 'objects_state_per_frame_with_time': objects_state_per_frame_with_time.view(self.max_sequence_of_frames*(self.n_nouns+4)),
                # 'objects_state_per_frame_with_time_noreshape': objects_state_per_frame_with_time
                } 

    def __len__(self):
        return len(self.action_table)

    def get_objects_per_frame(self, objects_df, video_segment, score_obj_position = False, center=None, state=False):

        objects_per_frame = torch.zeros((self.n_nouns, self.max_sequence_of_frames))
        for (frame, objects_group) in objects_df.groupby('frame'):
            frame_id = (frame - video_segment.start_frame) // 30
            objects_lst = list(objects_group.noun_class.get_values())

            if not score_obj_position:
                #0 no object, 1 object with out state, >1 object with state
                objects_per_frame[:self.n_nouns, frame_id] = self.to_categorial_multilabel(objects_lst, self.n_nouns)
                if state: 
                    state, object = self.get_object_states(video_segment, frame)
                    if object in objects_lst:
                        objects_per_frame[object, frame_id] = state+2
            else:
                for (noun_class, obj_df) in objects_group.groupby('noun_class'):
                    if center is None:
                        obj_scores = self.score_obj_on_position(obj_df)
                    else:
                        obj_scores = self.score_obj_on_hand(obj_df, center[:, frame_id])
                    objects_per_frame[noun_class, frame_id] = max(obj_scores)
                
        return objects_per_frame

    def get_gaussian_filter(self, g_size, center):
        from scipy.signal import gaussian
        return gaussian(g_size, center)

    def load_masks(self, file):
        import pickle, os.path
        if os.path.exists(self.masks_dir + file):
            with open(self.masks_dir + file, 'rb') as f:
                masks = pickle.load(f, encoding='latin-1')
            return (masks['segms'], masks['boxes'])
        return None, None

    def score_obj_on_hand(self, obj_df, hand_box):
        g_size = 100
        g_scale = g_size/456, g_size/256
        if (hand_box[2]-hand_box[0]) == 0 or (hand_box[3]-hand_box[1]) == 0: 
            print(hand_box)
        gaussian_hand_x = self.get_gaussian_filter(g_size,(hand_box[2]-hand_box[0])*g_scale[0])
        gaussian_hand_y = self.get_gaussian_filter(g_size,(hand_box[3]-hand_box[1])*g_scale[1])

        for (_, obj) in obj_df.iterrows():
            obj_bb = eval(obj.bounding_boxes)
            obj_score_position = [0]
            for bb in obj_bb:    
                norm_bb = [bb[0] * self.img_scale[1], 
                            bb[1]* self.img_scale[0], 
                            bb[2]* self.img_scale[1], 
                            bb[3]* self.img_scale[0]] #(top,left,height,width)
                center_bb = [int(norm_bb[2]//2), 
                            int(norm_bb[3]//2)]
                obj_score_position.append(gaussian_hand_x[int(center_bb[0]*g_scale[0])]*gaussian_hand_y[int(center_bb[1]*g_scale[1])])
        return obj_score_position

    def score_obj_on_position(self, obj_df):
        for (_, obj) in obj_df.iterrows():
            obj_bb = eval(obj.bounding_boxes)
            obj_score_position = [0]
            for bb in obj_bb:    
                norm_bb = [bb[0] * self.img_scale[1], 
                            bb[1]* self.img_scale[0], 
                            bb[2]* self.img_scale[1], 
                            bb[3]* self.img_scale[0]] #(top,left,height,width)
                center_bb = [int(norm_bb[2]//2), 
                            int(norm_bb[3]//2)]
                obj_score_position.append(self.gaussian_height[center_bb[0]]*self.gaussian_width[center_bb[1]])
        return obj_score_position



    def get_objects_in_segment(self, video_segment):
        object_table =  pd.read_pickle(self.annotation_dir + 'per_video/'+video_segment.video_id+'_objects.pkl')
        return object_table[(object_table.frame >= video_segment.start_frame) & (object_table.frame <= video_segment.stop_frame)]

    def get_guassian(self, sigma, mean, point):
        x, y = point
        x0, y0 = mean
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

    def get_cocohand_position_in_segment(self, video_segment):
        segms, boxes = self.load_masks(f'{video_segment.video_id}_{video_segment.start_timestamp}_{video_segment.stop_timestamp}.pkl')

        num_frames = video_segment.stop_frame - video_segment.start_frame+1
        hands_per_frame  = np.zeros((4, num_frames//2))

        if boxes is not None and len(boxes) > 0: 
            num_frames = len(boxes) 
            for frame_id, frame_nb in enumerate(range(0, num_frames,2)):
                obj_nb = 1 #human
                obj_box = boxes[frame_nb][obj_nb]
                hands_box = []
                for i in range(len(obj_box)):
                    size = segms[frame_nb][obj_nb][i]['size']
                    if obj_box[i] is not None and obj_box[i][-1] >= 0.1:
                        x1, y1, x2, y2, conf = boxes[frame_nb][obj_nb][i]
                        x1, x2 = x1 / self.img_scale[0], x2 / self.img_scale[0]
                        y1, y2 = y1 / self.img_scale[1], y2 / self.img_scale[1]
                        hands_box.append([x1, y1, x2, y2])
                        
                if len(hands_box) > 0: 
                    hands_box = np.mean(np.array(hands_box), axis=0)
                else:
                    hands_box = np.array([0, 0, 456, 256])
                hands_per_frame[:, frame_id] = hands_box
        else:
            hands_per_frame[2, :] = 456
            hands_per_frame[3, :] = 256
        return hands_per_frame

    def to_categorial_multilabel(self, labels, n_classes):
        '''returns a multilabeled vector (one at each class label) number of ones in the returned vector = len(labels)'''
        categorials = torch.zeros(n_classes)
        categorials[labels] = 1
        return categorials

    def pad_sequence(self, labels):
        padded_seq = -torch.ones(self.max_sequence_of_objects)
        padded_seq[:len(labels)] = torch.LongTensor(labels[:self.max_sequence_of_objects])
        return padded_seq
        

    def to_caterogrial(self, labels, n_classes):
        '''returns a mutlidimentional array (one vector for each class label) the diminsion of the array = (len(labels), n_classes)'''
        if len(labels) == 0:
            return torch.zeros(1, n_classes)
        categorials = torch.zeros(len(labels), n_classes)
        for i, l in enumerate(labels):
            categorials[i, l] = 1
        return categorials
        
    def get_timecounters_features(self, n_timeslots):
        counter_ascending  = [t for t in range(1, n_timeslots+1, +1)] 
        counter_descending = counter_ascending[::-1]
        counter_ascending_norm  = [t/(n_timeslots-1) for t in range(0, n_timeslots, +1)] if n_timeslots > 1 else [t for t in range(0, n_timeslots)]
        counter_descending_norm = counter_ascending_norm[::-1]

        padding =  [-1] * (self.max_sequence_of_frames - n_timeslots)
        return torch.stack([torch.Tensor(counter_ascending + padding), 
                            torch.Tensor(counter_descending + padding), 
                            torch.Tensor(counter_ascending_norm + padding), 
                            torch.Tensor(counter_descending_norm + padding) ])

    def get_object_states(self, segment, frame_id):
        verb2state_dict = self.verb2state_dict

        seg_v_class = str(segment.verb_class)
        state_dict  = verb2state_dict[seg_v_class] if seg_v_class in verb2state_dict else []
        num_frames = segment.stop_frame -segment.start_frame

        object_state=-1
        for s in state_dict:
            condition   = any(hint in s['hints']        for hint in segment.narration.split(' ') + [segment.noun]) or s['hints'] == []
            freeze_from = any(word in segment.narration for word in ['still', 'continue', 'continuing', 'stay'])
            cycle_state = (s['to'] == s['from'])
            
            if condition:
                if not freeze_from:
                    if frame_id < num_frames//2:
                        object_state = s['from']
                    else:
                        object_state = s['to']
                elif freeze_from:
                    object_state =  s['to']
                if cycle_state:    
                    object_state = s['to']

        return object_state, segment.noun_class

#test
'''home_db_dir = '/media/nachwa/48d9ff99-04f7-4a80-ae30-8bd5a89069f8/Datasets/epic_kitchen/'
work_db_dir = '/media/naboubak/Maxtor/EPIC_KITCHENS_2018/'
epic = EPIC_Dataset(work_db_dir, training=False)
print(epic.__getitem__(3))'''
