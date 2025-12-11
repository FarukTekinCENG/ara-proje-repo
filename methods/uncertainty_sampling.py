import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train import JobClassifierTrainer
from model import ModelPredictor

from data_utils.database import database

class ActiveLearning:
    hyper_params = {
        "N": 5,       # number of samples selected each iteration
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
                # get model answer
                single_result = predictor.predict(x[1])
                print(single_result["predicted_class"])
                print(single_result["confidence"])
                 
                # save model answer db
                # save score db
                #print(x)
                uncertainty=1-single_result["confidence"]
                database.save_model_prediction(
                    sample_id=x[0],
                    predicted_class=single_result["predicted_class"],
                    uncertainty_score=uncertainty
                )                

    @staticmethod
    def prep_labels(samples):
        # for real scenarios > at this stage: label unlabelled data
        # manual labelling or ai supported labelling
        print("now label your suggested samples...")
        # (UPDATE is_labelled TRUE) AND (UPDATE LABEL [both in dataset and pool])
        return samples

    @staticmethod
    def train_iterate(samples, labels=None):
        trainer = JobClassifierTrainer()
        trainer.pipeline(samples, labels)

    @staticmethod
    def check_stop_condition():
        ...
        # until:
        # (no unlabelled sample left in pool) V
        # (no imporovements achieved better than I in last 2 rounds of training) V
        # (model trust is above T Threshold)
        return 'met'

    @staticmethod
    def uncertainty_sampling(max_samples=None):
        condition='unmet'

        while condition=='unmet':
            # let model make predictions
            ActiveLearning.model_predict(max_samples)
            
            # select samples with Uncertainty Sampling
            selected_samples = database.select_samples_to_train(ActiveLearning.hyper_params["N"])
            #print(selected_samples)
            
            # label selected samples
            selected_samples_labelled = ActiveLearning.prep_labels(selected_samples)
            
            # re-train with labelled data
            ActiveLearning.train_iterate(selected_samples_labelled)
            
            # check stop condition
            condition = ActiveLearning.check_stop_condition()
        
        # print("success: ... rounds of training: ... max labels: ...")

if __name__ == '__main__':
    ActiveLearning.uncertainty_sampling(50)

