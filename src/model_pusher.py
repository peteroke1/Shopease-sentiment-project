import mlflow
import mlflow.transformers
import dagshub
import logging
from utils.model_utils import get_best_f1
from config.constant import model_name, training_args
from transformers import pipeline

logging.basicConfig(
    level = logging.DEBUG,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)


class model_pusher:
    def __init__(self, experiment_name="sentiment_analysis_experiment"):
        
       try:
           dagshub.init(repo_owner= "peteroke1", repo_name= "Shopease-sentiment-project", mlflow=True)
           self.experiment_name = experiment_name
           mlflow.set_experiment(experiment_name)
           logging.info(f"Model pusher has been successfully initialized..")
       except Exception as e:
           logging.error("error occurred while initializing model_pusher {e}")

    def updated_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics["eval_f1"]
            old_f1 = get_best_f1(self.experiment_name)

            print(f"New f1 score: {new_f1}")
            print("f Old f1 score: {old_f1}")

            if old_f1 is None or new_f1 > old_f1:
                with mlflow.start_run():

                    #log the metrics
                    mlflow.log_metrics("accuracy", metrics["eval_accuracy"])
                    mlflow.log_metrics("fi", new_f1)

                    #log the parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("eval_batch_size", training_args.per_device_train_batch_size)

                    # Create a pipeline
                    sentiment_pipeline = pipeline(
                        task = "task-classification",
                        model = trainer.model,
                        tokenizer = model_name,
                        return_all_score = True
                    )

                    #log the model with the tokenizer
                    mlflow.transformers.log_model(
                        artifact_path = model,
                        registered_model_name = model_name
                    )
                logging.info(f"Model and metrics have been successfully pushed to mlflow..")
            else:
                logging.info(f"Model not pushed to mlflow, performance of the model didn't improve..")

        except Exception as e:
            logging.error(f"failed to push model and result to mlflow {e}")