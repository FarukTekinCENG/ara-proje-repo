import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import train
from model import ModelPredictor

from data_utils.database import database

class ActiveLearning:
    hyper_params = {
        "N": 100,       # number of samples selected each iteration
        "I": 0.001,     # ...
        "T": 0.85       # model prediction certainty threshold
    }
    
    @staticmethod
    def model_predict(max_samples=None):
        # 
        predictor = ModelPredictor("./fine_tuned_eurobert")

        # SELECT is_labelled false samples in pool
        batch_size = 1000
        offset = 0
        total_processed = 0

        # pagination
        page = 0
        while True:
            if max_samples is not None and total_processed >= max_samples:
                print(f"Max samples limit reached: {max_samples}")            
                break
            
            offset = page * batch_size
            batch=database.get_unlabelled_samples(batch_size, offset)
            

            if not batch:
                break
                
            # Eğer max_samples var, son batch'i kırp
            if max_samples is not None:
                remaining = max_samples - total_processed
                if len(batch) > remaining:
                    batch = batch[:remaining]
            
            #print(batch)        
            print(f"Page {page}: {len(batch)} records")
            total_processed += len(batch)
            page += 1

            for x in batch:
                # make model predict
                # load model, give input, take output
                single_result = predictor.predict(x[1])
                print(single_result["predicted_class"])
                print(single_result["confidence"])
                 
                # save model answer db
                
                # calc score
                #uncertainty_score=calc_score()
                uncertainty_score=1-single_result["confidence"]

                # save score db
                #save_score_db(uncertainty_score)

    # @staticmethod
    # def calc_score():
    #     # ...
    #     # calc score
    #

    # @staticmethod
    # def select_samples_to_train(N=100):
    #     # ...
    #

    # @staticmethod
    # def prep_labels():
    #     # ...
    #     # for real scenarios > at this stage: label unlabelled data
    #

    # @staticmethod
    # def train_iterate(samples, labels):
    #     train.pipeline(samples, labels)
    #
    # @staticmethod
    # def check_stop_condition():
    #     ...
    #     # until:
    #     # (no unlabelled sample left in pool) V
    #     # (no imporovements achieved better than I in last 2 rounds of training) V
    #     # (model trust is above T Threshold)

    @staticmethod
    def run(max_samples=None):
        # TEST: Sadece 1 iterasyon çalıştır
        print("Running single iteration for testing...")
        UncertaintySampling.model_predict(max_samples)
        print("Test iteration completed")        

        # ESAS KOD
        # condition='unmet'
        #
        # while condition=='unmet':
        #     UncertaintySampling.model_predict(max_samples)
        #     # calc_scores()
        #     # samples=select_samples_to_train(hyper_params["N"])
        #     # labels=prep_labels(samples)
        #     # train_iterate(samples, labels)
        #     # condition=check_stop_condition()
        #
        # print("success: ... rounds of training: ... max labels: ...")

if __name__ == '__main__':
    ActiveLearning.run(5)

