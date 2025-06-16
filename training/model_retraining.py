import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras.callbacks import EarlyStopping
from upload_model import DagsHubMLflowLogger
import pandas as pd 
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)

repo_name = "Corix"
repo_owner = "slalrijo2005"

Dagshub = DagsHubMLflowLogger(repo_name=repo_name,repo_owner=repo_owner)

class Mixin:
    
    def __init__(self,threshold: float,factor: float): 
        """
        Base mixin class providing utilities for handling class imbalance and outliers.

        Args:
            threshold: Imbalance ratio threshold for applying SMOTE.
            factor: Outlier removal multiplier for IQR.
        """
        self.threshold = threshold
        self.factor = factor
        
    def balance_dataset(self,x_train: DataFrame,y_train: Series) -> tuple[DataFrame,Series]: 
        """
        Applies SMOTE oversampling if imbalance exceeds threshold.

        Args:
            x_train: Training features.
            y_train: Training labels.

        Returns:
            Resampled x_train and y_train.
        """
        try:
            class_counts = Counter(y_train)
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())

            imbalance_ratio = max_count / min_count

            if imbalance_ratio > self.threshold:
                logger.info(f"Dataset is unbalanced with imbalance ratio of {imbalance_ratio}")
                sampler = SMOTE(random_state=42)
                x_res, y_res = sampler.fit_resample(x_train, y_train)
            else:
                logger.info(f"Dataset is balanced. Proceeding without resampling")
                x_res, y_res = x_train, y_train
            
            return x_res, y_res
        
        except Exception as e:
            logger.error(f"Balancing dataset failed due to {e}")
        
    def outliers(self,series: DataFrame) -> DataFrame:
        """
        Clipping outliers in each feature column based on IQR method.

        Args:
            series: A DataFrame to clip outliers.

        Returns:
            Clipped DataFrame.
        """
        try:
            
            logger.info(f"Handling outliers")
            
            q1 = np.percentile(series, 25)
            q3 = np.percentile(series, 75)
            iqr = q3 - q1

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr 
            
            return series.clip(lower, upper)
        
        except Exception as e:

            logger.error(f"Outliers removal failed due to {e}")

class Lab(Mixin):
    def __init__(self,dataset,threshold: float = 1.5,factor:float = 1.5):
        
        """
        Handles deep learning lab model lifecycle including data prep, training, and evaluation.

        Args:
            dataset: Input dataset.
            threshold: Imbalance threshold.
            factor: IQR factor for outlier handling.
        """
        super().__init__(threshold,factor)
        self.dataset = dataset
        self.model_optimizer = Adam(learning_rate=0.001)
        self.model_loss = BinaryCrossentropy()
        self.model_accuracy = 'accuracy' 
        self.model_monitor = 'val_accuracy'
        self.model_selection_thresh = 0.6
        self.scaler = StandardScaler()

    def split_data(self,test_size:float,target_name:str) -> tuple[DataFrame,DataFrame,Series,Series]:
        """
        Splits dataset into training and test sets.

        Args:
            test_size: Proportion of test data.
            target_name: Target column name.

        Returns:
            Tuple of training and test splits.
        """
        try:
            X = self.dataset.drop(target_name,axis=1)
            Y = self.dataset[target_name]
            
            X = self.outliers(X)
            
            x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=24,
                                                                test_size=test_size,
                                                                shuffle=True)
            logger.info("Lab dataset split successful")
            
            return x_train, x_test, y_train, y_test 
        except Exception as e:
            logger.error(f"Error while splitting Lab dataset: {e}")
        
    def scaling(self,x_train:DataFrame,x_test:DataFrame) -> tuple[np.ndarray,np.ndarray]:
        """
        Scales feature data using standard scaler.

        Returns:
            Scaled training and test data.
        """
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        return x_train_scaled, x_test_scaled
    
    def preprocessing(self,target_name:str, test_size:float=0.2) -> tuple[np.ndarray, np.ndarray, Series, Series]:
        """
        Prepares data: splits, balances, and scales.

        Returns:
            Final training and test sets.
        """
        x_train, x_test, y_train, y_test = self.split_data(test_size,target_name)
        x_train, y_train = self.balance_dataset(x_train,y_train)
        x_train, x_test = self.scaling(x_train,x_test)
        return x_train,x_test,y_train,y_test 
    
    def is_model_good(self,x_test:np.ndarray,y_test:Series,model:Sequential):
        """
        Evaluates model against test set.

        Returns:
            Loss and accuracy.
        """
        eval_loss,eval_accuracy = model.evaluate(x_test,y_test)
        if eval_accuracy > self.model_selection_thresh:
             logger.info(
                f"Lab model meets performance criteria — Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy * 100:.2f}%"
            )
        else:
            logger.warning(
                f"Lab model does NOT meet performance criteria — Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy * 100:.2f}%"
            )
        return eval_loss, eval_accuracy
    
    def make_model(self,input_shape:tuple[int]) -> Sequential:
        """
        Builds a Keras deep learning model.

        Returns:
            Compiled Keras model.
        """
        model = Sequential([
            tf.keras.Input(shape=input_shape),  
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

       
        model.compile(
            optimizer=self.model_optimizer,
            loss=self.model_loss,
            metrics=[self.model_accuracy]
        )
        return model 
    
    def training(self,x_train:np.ndarray, y_train:Series, model:Sequential) -> None:
        """
        Trains the Keras model using early stopping.
        """
        early_stop = EarlyStopping(
                        monitor=self.model_monitor,   
                        patience=5,           
                        restore_best_weights=True
                    )
        model.fit(
            x_train, y_train,
            epochs=20,
            batch_size=32,
            verbose=1,
            callbacks=[early_stop]
        )
        
    
    def upload_model_to_mlflow(self,model:Sequential,eval_loss:float,eval_acc:float) -> None:
        """
        Logs the trained TensorFlow model to DagsHub MLflow.
        """
        Dagshub.log_tensorflow_model(model,
                                     eval_loss,
                                     eval_acc,
                                    )

class Wearable(Mixin):
    def __init__(self,dataset: DataFrame,threshold:float=1.5,factor:float=1.5):
        """
        Handles classical ML model lifecycle for wearable dataset.

        Args:
            dataset: Input dataset.
            threshold: Imbalance threshold.
            factor: Outlier threshold factor.
        """
        super().__init__(threshold,factor)
        self.dataset = dataset
        self.model = RandomForestClassifier()
        self.param_dist = {
            "n_estimators": np.random.randint(100, 200, 5).tolist(),
            "max_depth": np.random.randint(5, 30, 5).tolist(),
            "min_samples_split": np.random.randint(2, 15, 5).tolist(),
            "criterion": ["gini", "entropy"]
        }
        self.scaler = StandardScaler()
        self.MIN_ACCURACY = 0.90
        self.MIN_F1 = 0.85
        self.n_iter = 10
        self.cv = 3
        
    
    def split_data(self,test_size:float,target_name:str) -> tuple[DataFrame, DataFrame, Series, Series]:
        """
        Splits the wearable dataset.

        Returns:
            Training and test sets.
        """
        try:
            X = self.dataset.drop(target_name,axis=1)
            Y = self.dataset[target_name]
            
            X = self.outliers(X)
            
            x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=24,
                                                                test_size=test_size,
                                                                shuffle=True)
            logger.info("Wearable dataset split successful")
            
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error while splitting wearable dataset: {e}")
            
    def preprocessing(self,target_name:str,test_size:float=0.2) -> tuple[DataFrame, DataFrame, Series, Series]:
        """
        Preprocesses wearable data.

        Returns:
            Cleaned and split dataset.
        """
        x_train, x_test, y_train, y_test = self.split_data(test_size,target_name)
        x_train, y_train = self.balance_dataset(x_train,y_train)
        return x_train,x_test,y_train,y_test
    
    def hyperparameter_tuning(self,x_train:DataFrame,y_train:Series) -> dict:
        """
        Performs randomized search for best model hyperparameters.

        Returns:
            Best hyperparameters found.
        """
        rfc = RandomizedSearchCV(
            self.model,
            self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
        )
        rfc.fit(x_train,y_train)
        
        return rfc.best_params_
 
        
    def is_model_good(self,x_test:DataFrame,y_test:Series,trained_model:BaseEstimator) -> tuple[float, float]:
        """
        Evaluates the trained sklearn model.

        Returns:
            Accuracy and F1 score.
        """
        
        y_pred = trained_model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        if accuracy >= self.MIN_ACCURACY and f1 >= self.MIN_F1:
            logger.info(
                f"Lab model meets performance criteria — Accuracy: {accuracy}, F1: {f1}%"
            )
        else:
            logger.warning(
                f"Lab model does NOT meet performance criteria — Accuracy: {accuracy}, F1: {f1}%"
            )
        return accuracy,f1
    
    def training(self,x_train:DataFrame,y_train:Series) -> BaseEstimator:
        """
        Trains a RandomForestClassifier with the best hyperparameters.

        Returns:
            Trained sklearn model.
        """
        best_param = self.hyperparameter_tuning(x_train,y_train)
        self.best_param = best_param

        random_forest_model = RandomForestClassifier(**best_param)
        random_forest_model.fit(x_train,y_train)
        return random_forest_model
    
    def upload_model_to_mlflow(self,model:BaseEstimator, accuracy:float, f1_score:float) -> None:
        """
        Logs sklearn model and its metrics to MLflow via DagsHub.
        """
        Dagshub.log_sklearn_model(model,
                                 self.best_param,
                                 accuracy,
                                 f1_score,    
                                 )
        
     
    
data = pd.read_csv("training_datasets/cardiac_signals_wearables.csv")
wearable = Wearable(data)
x_train,x_test,y_train,y_test = wearable.preprocessing("HeartDisease")
model = wearable.training(x_train,y_train)
accuracy,f1 = wearable.is_model_good(x_test,y_test,model)
wearable.upload_model_to_mlflow(model,accuracy,f1)

data = pd.read_csv("training_datasets/cardio_lab_train.csv")
lab = Lab(data)
x_train,x_test,y_train,y_test = lab.preprocessing("cardio")
model = lab.make_model((12,))
lab.training(x_train,y_train,model)
eval_loss, eval_accuracy = lab.is_model_good(x_test,y_test,model)
lab.upload_model_to_mlflow(model,eval_loss,eval_accuracy)