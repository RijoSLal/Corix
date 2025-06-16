import mlflow


class Model_Fetch:
    def __init__(self,tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)

    def load_lab_model(self,model_name,model_version):
        
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.tensorflow.load_model(model_uri)
        return model 

    def load_wearable_model(self,model_name,model_version):

        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri)
        return model 
