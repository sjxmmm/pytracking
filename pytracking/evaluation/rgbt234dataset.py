import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob
import pdb

def RGBT234Dataset():
    return RGBT234DatasetClass().get_sequence_list()


class RGBT234DatasetClass(BaseDataset):
    """VOTRGBT2018 dataset

    Publication:
        The sixth Visual Object Tracking VOTRGBT2018 challenge results.
        Matej Kristan, Ales Leonardis, Jiri Matas, Michael Felsberg, Roman Pfugfelder, Luka Cehovin Zajc, Tomas Vojir,
        Goutam Bhat, Alan Lukezic et al.
        ECCV, 2018
        https://prints.vicos.si/publications/365

    Download the dataset from http://www.votchallenge.net/vot2019rgbt/dataset.html"""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbt234_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name        

        anno_path = '{}/{}/visible.txt'.format(self.base_path, sequence_name)   # or visible infrared
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
        end_frame = ground_truth_rect.shape[0]
        anno_pathi = '{}/{}/infrared.txt'.format(self.base_path, sequence_name)   # or visible infrared
        try:
            ground_truth_recti = np.loadtxt(str(anno_pathi), dtype=np.float64)
        except:
            ground_truth_recti = np.loadtxt(str(anno_pathi), delimiter=',', dtype=np.float64)
        end_framei = ground_truth_recti.shape[0]

        img_dir = os.path.join(self.base_path, sequence_path, 'visible')   
        img_list = glob.glob(img_dir + "/*.jpg")
        img_list.sort()
        frames = [os.path.join(img_dir, x) for x in img_list]
        imgi_dir = os.path.join(self.base_path, sequence_path, 'infrared')   
        imgi_list = glob.glob(imgi_dir + "/*.jpg")
        imgi_list.sort()
        framesi = [os.path.join(imgi_dir, x) for x in imgi_list]        
#        return Sequence(sequence_name, framesv, ground_truth_rect)
        return Sequence(sequence_name, frames, framesi, 'RGBT234', ground_truth_rect, ground_truth_recti)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['afterrain',
                         'aftertree',
                         'baby',
                         'baginhand',
                         'baketballwaliking',
                         'balancebike',
                         'basketball2',
                         'bicyclecity',
                         'bike',
                         'bikeman',
                         'bikemove1',
                         'biketwo',
                         'blackwoman',
                         'blueCar',
                         'bluebike',
                         'boundaryandfast',
                         'bus6',
                         'call',
                         'car',
                         'car10',
                         'car20',
                         'car3',
                         'car37',
                         'car4',
                         'car41',
                         'car66',
                         'carLight',
                         'caraftertree',
                         'carnotfar',
                         'carnotmove',
                         'carred',
                         'child',
                         'child1',
                         'child3',
                         'child4',
                         'children2',
                         'children3',
                         'children4',
                         'crossroad',
                         'crouch',
                         'cycle1',
                         'cycle2',
                         'cycle3',
                         'cycle4',
                         'cycle5',
                         'diamond',
                         'dog',
                         'dog1',
                         'dog10',
                         'dog11',
                         'elecbike',
                         'elecbike10',
                         'elecbike2',
                         'elecbike3',
                         'elecbikechange2',
                         'elecbikeinfrontcar',
                         'elecbikewithhat',
                         'elecbikewithlight',
                         'elecbikewithlight1',
                         'face1',
                         'floor-1',
                         'flower1',
                         'flower2',
                         'fog',
                         'fog6',
                         'glass',
                         'glass2',
                         'graycar2',
                         'green',
                         'greentruck',
                         'greyman',
                         'greywoman',
                         'guidepost',
                         'hotglass',
                         'hotkettle',
                         'inglassandmobile',
                         'jump',
                         'kettle',
                         'kite2',
                         'kite4',
                         'luggage',
                         'man2',
                         'man22',
                         'man23',
                         'man24',
                         'man26',
                         'man28',
                         'man29',
                         'man3',
                         'man4',
                         'man45',
                         'man5',
                         'man55',
                         'man68',
                         'man69',
                         'man7',
                         'man8',
                         'man88',
                         'man9',
                         'manafterrain',
                         'mancross',
                         'mancross1',
                         'mancrossandup',
                         'mandrivecar',
                         'manfaraway',
                         'maninblack',
                         'maninglass',
                         'maningreen2',
                         'maninred',
                         'manlight',
                         'manoccpart',
                         'manonboundary',
                         'manonelecbike',
                         'manontricycle',
                         'manout2',
                         'manup',
                         'manwithbag',
                         'manwithbag4',
                         'manwithbasketball',
                         'manwithluggage',
                         'manwithumbrella',
                         'manypeople',
                         'manypeople1',
                         'manypeople2',
                         'mobile',
                         'night2',
                         'nightcar',
                         'nightrun',
                         'nightthreepeople',
                         'notmove',
                         'oldman',
                         'oldman2',
                         'oldwoman',
                         'orangeman1',
                         'people',
                         'people1',
                         'people3',
                         'playsoccer',
                         'push',
                         'rainingwaliking',
                         'raningcar',
                         'redbag',
                         'redcar',
                         'redcar2',
                         'redmanchange',
                         'rmo',
                         'run',
                         'run1',
                         'run2',
                         'scooter',
                         'shake',
                         'shoeslight',
                         'single1',
                         'single3',
                         'soccer',
                         'soccer2',
                         'soccerinhand',
                         'straw',
                         'stroller',
                         'supbus',
                         'supbus2',
                         'takeout',
                         'tallman',
                         'threeman',
                         'threeman2',
                         'threepeople',
                         'threewoman2',
                         'together',
                         'toy1',
                         'toy3',
                         'toy4',
                         'tree2',
                         'tree3',
                         'tree5',
                         'trees',
                         'tricycle',
                         'tricycle1',
                         'tricycle2',
                         'tricycle6',
                         'tricycle9',
                         'tricyclefaraway',
                         'tricycletwo',
                         'twoelecbike',
                         'twoelecbike1',
                         'twoman',
                         'twoman1',
                         'twoman2',
                         'twoperson',
                         'twowoman',
                         'twowoman1',
                         'walking40',
                         'walking41',
                         'walkingman',
                         'walkingman1',
                         'walkingman12',
                         'walkingman20',
                         'walkingman41',
                         'walkingmantiny',
                         'walkingnight',
                         'walkingtogether',
                         'walkingtogether1',
                         'walkingtogetherright',
                         'walkingwithbag1',
                         'walkingwithbag2',
                         'walkingwoman',
                         'whitebag',
                         'whitecar',
                         'whitecar3',
                         'whitecar4',
                         'whitecarafterrain',
                         'whiteman1',
                         'whitesuv',
                         'woamn46',
                         'woamnwithbike',
                         'woman',
                         'woman1',
                         'woman100',
                         'woman2',
                         'woman3',
                         'woman4',
                         'woman48',
                         'woman6',
                         'woman89',
                         'woman96',
                         'woman99',
                         'womancross',
                         'womanfaraway',
                         'womaninblackwithbike',
                         'womanleft',
                         'womanpink',
                         'womanred',
                         'womanrun',
                         'womanwithbag6',
                         'yellowcar']

        return sequence_list
