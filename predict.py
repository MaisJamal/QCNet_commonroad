# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet

import datasets.cr_extractor as extractor
import datasets.cr_argoverse_converter as conv
from commonroad.common.file_reader import CommonRoadFileReader
import time

import os, shutil
import yaml

with open('ffstreams/config/config.yml', 'r') as file:
    config = yaml.safe_load(file)

def predict_traj():
    if os.path.isdir("test_test/test"):
        shutil.rmtree("test_test/test")
    pl.seed_everything(2023, workers=True)

    # parser = ArgumentParser()
    # parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--root', type=str, required=True)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--pin_memory', type=bool, default=True)
    # parser.add_argument('--persistent_workers', type=bool, default=True)
    # parser.add_argument('--accelerator', type=str, default='auto')
    # parser.add_argument('--devices', type=int, default=1)
    # parser.add_argument('--ckpt_path', type=str, required=True)
    # args = parser.parse_args()
    ## set parameters
    root_arg = "/home/mais2/QCNet/test_test/"
    ckpt_path_arg = "/home/mais2/QCNet/lightning_logs/version_9/checkpoints/epoch=0-step=199908.ckpt"
    model_arg = "QCNet"
    batch_size_arg = 32
    num_workers_arg = 8
    pin_memory_arg = True
    persistent_workers_arg = True
    devices_arg = 1
    accelerator_arg = 'auto'
    #####
    model = {
        'QCNet': QCNet,
    }[model_arg].load_from_checkpoint(checkpoint_path=ckpt_path_arg)
    test_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=root_arg, split='test')
    dataloader = DataLoader(test_dataset, batch_size=batch_size_arg, shuffle=False, num_workers=num_workers_arg,
                            pin_memory=pin_memory_arg, persistent_workers=persistent_workers_arg)
    trainer = pl.Trainer(accelerator=accelerator_arg, devices=devices_arg, strategy='ddp')
   # predictions = trainer.test(model, dataloader)
    pred = trainer.predict(model, dataloader,return_predictions=True)
    pred_traj = pred[0][0]
    pred_prob = pred[0][1]
    print("pred" ,pred)
    print("pred traj shape" ,pred_traj.shape)
    print("pred probabilities shape" ,pred_prob.shape)
    return pred_traj,pred_prob


if __name__ == '__main__':

    #scene_path = "datasets/commonroad/USA_US101-1_1_T-1.xml"     #change_scenario
    #scene_path = "datasets/commonroad/DEU_Nuremberg-39_5_T-1.xml"    
    #data = extractor.cr_scene_to_qcnet(scene_path)
    #scenario, planning_problem_set = CommonRoadFileReader(scene_path).open()
    #argo_map,centerlines = conv.converter(scenario, planning_problem_set)
    if config['prediction_visualization']['visualize'] == False:
        print ("visualization off")
    if os.path.isdir("test_test/test"):
        shutil.rmtree("test_test/test")
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    args = parser.parse_args()

    model = {
        'QCNet': QCNet,
    }[args.model].load_from_checkpoint(checkpoint_path=args.ckpt_path)
    test_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='test')
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp')
   # predictions = trainer.test(model, dataloader)
    pred = trainer.predict(model, dataloader,return_predictions=True)

    print("pred" ,pred)
    print("pred traj shape" ,pred[0][0].shape)
    print("pred probabilities shape" ,pred[0][1].shape)
    # second cycle
    time_start = time.time()
    if os.path.isdir("test_test/test"):
        shutil.rmtree("test_test/test")
    test_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split='test')
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    pred2 = trainer.predict(model, dataloader,return_predictions=True)

    print("pred" ,pred2)
    print("cycle duration : ", time.time()-time_start)
 


   # 'loc_refine_pos': loc_refine_pos,
   #         'scale_refine_pos': scale_refine_pos,
   #         'loc_refine_head': loc_refine_head,
     #       'conc_refine_head': conc_refine_head,