import argparse
import os
from pathlib import Path


import data as data_package

import models as model_package
import numpy as np
import torch
import yaml  # type: ignore
from torch.utils.data import DataLoader
from utils import PerformanceEvaluator, init_obj
from typing import List
from collections import defaultdict
from scipy import stats
from collections import Counter

'''
Run Inference, support for ensemble models
Generate Result for each model, then aggregate it to give final result 
'''

        
class ConfigArgs(object):
    def __init__(self, **arg_dicts) -> None:
        self.__dict__.update(arg_dicts)


class Test():
    def __init__(
        self, 
        args: "argparse.Namespace",  
    ) -> None:
        self.args= args
        self.init_test()
        self.init_utils()


    def init_utils(self):
        self.test_evaluator = PerformanceEvaluator()


    def init_test(self):
        """ 
        initialize configs for model testing
        load test dataset
        """

        # load dataset 
        self.collate_function = init_obj(self.args.collate_function, data_package)
        self.test_dataset = init_obj(self.args.test_dataset, data_package)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size = self.args.batch_size, num_workers = 0, shuffle = False, collate_fn = self.collate_function, drop_last=False)


    def get_save_path(self, model_path):
        '''
        create result folder
        '''
        save_root = Path(model_path).parent / self.args.test_dataset['args']['type']
        save_root.mkdir(parents=True, exist_ok=True)
        return save_root


    def cal_and_print_metrics(self):
        
        if len(Counter(self.label_all)) <= 1:
            return

        self.test_evaluator.cal_performance()
        print(f'Test Set AUC {self.test_evaluator.auc:.4f} | accu {self.test_evaluator.accu:.4f}')
        print(f'MCC :{self.test_evaluator.MCC:.4f} | F1_Score:{self.test_evaluator.F1_score:.4f}')
        print(f'Test Sensitivity:{self.test_evaluator.sensitivity:.4f} | Specificity:{self.test_evaluator.specificity:.4f}')


    def test(self):

        self.res_all_all_models = []
        
        for single_model_path in self.args.model_path:

            save_template = self.test_dataset.data.copy()
            self.init_utils()

            self.model = init_obj(self.args.model, model_package)
            self.model.load_state_dict(torch.load(single_model_path))
            self.model = self.model.cuda()
            self.model.eval()
            print('*'*10, f'Loading Model {single_model_path}', '*'*10)
            print('*'*10, 'Running Inference Start', '*'*10)
        

            self.res_all = []
            self.peptide_all = []
            self.label_all = []
            self.hla_type_all = []


            for batch, data in enumerate(self.test_dataloader):
                if torch.cuda.is_available():
                    data = {k: v.cuda() if isinstance(v, torch.Tensor) \
                            else v \
                            for k, v in data.items()}

                output = self.model(**data)
                prob =  torch.softmax(output, dim=-1)[:, 1]
                res = prob.cpu().detach().numpy()
                label = data["label"].cpu().detach().numpy()
                
                self.res_all.extend(res)
                self.hla_type_all.extend(data['hla_type'])
                self.peptide_all.extend(data['epitope'])
                self.label_all.extend(label)
                self.test_evaluator.update(label, res)   

            save_template.insert(loc = 1, column = 'model_score', value = self.res_all)
            save_path = os.path.join(self.get_save_path(single_model_path), 'model_score.csv')
            save_template.to_csv(save_path)

            print('*'*10, f'Save Result to {save_path}', '*'*10)

            # calculate correlation between binding/presentation score and immunogenic label 
            # corr = stats.pointbiserialr(self.label_all, self.res_all)
            # print(f'correlation {corr}')
            
            self.cal_and_print_metrics()
            self.res_all_all_models.append(self.res_all)
        
            print('*'*10, 'Running Inference End', '*'*10)
            #print('\n')


        if len(self.args.model_path) > 1:
            print('*'*10, 'Calculating Model Ensemble Performance', '*'*10)

            self.init_utils()
            self.ensemble_score = np.mean(np.array(self.res_all_all_models), axis = 0)
            #print(self.ensemble_score)
            assert len(self.ensemble_score) == len(self.label_all)
            self.test_evaluator.update(self.label_all, self.ensemble_score)   
            self.cal_and_print_metrics()
            
            save_template = self.test_dataset.data.copy()
            save_template.insert(loc = 1, column = 'model_score', value = self.ensemble_score)
            save_path = self.get_save_path(self.args.model_path[0].parent) / 'model_score.csv'
            save_template.to_csv(save_path)

        

        
        
def parser_args() -> ConfigArgs:
    parser = argparse.ArgumentParser(description="torchepitope testing")

    parser.add_argument("-p", "--path", type=str, help="models_saved_dir_path, contains yaml and .pth file")
    parser.add_argument("-m", "--model", default='best_model.pth', type=str, help="model_name")
    parser.add_argument("-e", "--ensemble", default=False, type=bool, help="aggregate predictions from multiple models")


    parser.add_argument("-t", "--test_dataset", default='', type=str, help="test_benchmark")



    args = parser.parse_args()
    model_path_list = []
    if not args.ensemble:
        config_path = Path(args.path) / 'config.yaml'
        assert config_path.is_file(), print('Please input a valid path which contains config.yaml file')
        model_path = Path(args.path) / args.model
        assert model_path.is_file(), print('Model doesn\'t exist, check again.')
        with open(config_path, "rt") as f:
                config_dict = yaml.safe_load(f)
        model_path_list.append(model_path)
    else:
        model_dir_list = os.listdir(Path(args.path))
        for model in model_dir_list:
            config_path = Path(args.path) / Path(model) / 'config.yaml'
            if not config_path.is_file():
                continue
            model_path = Path(args.path) / Path(model) / args.model
            if not model_path.is_file():
                continue

            with open(config_path, "rt") as f:
                config_dict = yaml.safe_load(f)
            model_path_list.append(model_path)
            

    # load base setting

    config_args = ConfigArgs(**config_dict)
    # update test setting
    config_args.test_dataset['args']['type'] = args.test_dataset 
    config_args.model_path = model_path_list

    

    return config_args


def main():
    args = parser_args()
    tester = Test(args=args)
    tester.test()
            
    
if __name__ == "__main__":
    main()
