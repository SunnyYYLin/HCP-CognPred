import torch
from config import *
from model.behav_pred import BehaviorPredictionModel
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# Set the config
with open("data/prediction_variables.txt") as f:
    pred_vars = f.read().splitlines()
'''backbone_config = PartialLinearRegressionConfig(
    hidden_dims=[64],
    dropout=0.0,
)'''
def init_config(backbone_config):
    config = PipelineConfig(
        input_dim=79800,
        pred_vars=pred_vars,
        r2loss_weight=0,
        backbone_config=backbone_config
    )
    config.backbone_config.input_dim = config.input_dim
    config.backbone_config.target_dim = config.target_dim
    return config
safetensors:str
def sensitivity_analysis(config, model_pth , top_k, input):
    """
        Perform sensitivity analysis on a given model and input data.
        This function loads a pre-trained model, performs a forward pass with the input data,
        computes the gradients of the predictions with respect to the input, and identifies
        the top-k features with the highest sensitivity.
        Args:
            config (dict): Configuration dictionary for the model.
            model_pth (.safetensors): Path to the pre-trained model file
            top_k (int): Number of top features to return based on sensitivity.
            input (torch.Tensor): Input data for the model.
        Returns:
            dict: A dictionary where keys are the indices of the top-k features and values are their corresponding sensitivity values.
    """

    model = BehaviorPredictionModel(config)
    print(model)
    
    state_dict = load_file(model_pth)
    model.load_state_dict(state_dict)
    
    model.eval()
    
    input_sample = input
    input_sample.requires_grad = True
    output = model(input_sample)

    predictions = output['predictions']

    predictions.backward(torch.ones_like(predictions))
    gradients = input_sample.grad

    sensitivity = gradients.abs().numpy()

    sensitivity_sorted = sensitivity.argsort()[::-1]
    top_k_features = sensitivity_sorted[:top_k]
    top_k_sensitivity = sensitivity[top_k_features]
    
    return {index: sensitivity for index, sensitivity in zip(top_k_features, top_k_sensitivity)}, sensitivity

def show_sensitive_features(config,model_pth,top_k,input):
    """
        Perform sensitivity analysis on a given model and input data.
        This function loads a pre-trained model, performs a forward pass with the input data,
        computes the gradients of the predictions with respect to the input, and identifies
        the top-k features with the highest sensitivity.
        Args:
            config (dict): Configuration dictionary for the model.
            model_pth (.safetensors): Path to the pre-trained model file
            top_k (int): Number of top features to return based on sensitivity.
            input (torch.Tensor): Input data for the model.
        Returns:
            dict: A dictionary where keys are the indices of the top-k features and values are their corresponding sensitivity values.
    """
    top_k_features,sensitivity = sensitivity_analysis(config, model_pth, top_k, input)
    features = list(range(1, top_k + 1))
    sensitivities = list(top_k_features.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(features, sensitivities, marker='o')
    plt.xlabel('Feature Rank')
    plt.ylabel('Sensitivity')
    plt.title('Top K Sensitive Features')
    plt.grid(True)
    plt.show()
    return top_k_features,sensitivity
if __name__ == '__main__':
    input_path='data\mean_GSR.npy'
    input=torch.tensor(np.loadtxt(input_path))
    model_pth='checkpoints\mlp_hidden[256,64]_dropout0.1_r2w0\checkpoint-638\model.safetensors'
    top_k=79800
    backbone_config=MLPConfig(hidden_dims=[256,64], dropout=0.1)
    config = init_config(backbone_config)
    top_k_features,sensitivity=show_sensitive_features(config,model_pth,top_k,input)
    #保存结果成文件,还原成400*400adj矩阵
    np.savetxt('data\sensitivity.npy', sensitivity)
    sensitivity = np.loadtxt('data\sensitivity.npy')
    sensitivity_matrix = np.zeros((400, 400))
    upper_triangle_indices = np.triu_indices(400, k=1)
    sensitivity_matrix[upper_triangle_indices] = sensitivity
    sensitivity_matrix = sensitivity_matrix + sensitivity_matrix.T
    sensitivity = sensitivity_matrix
    np.savetxt('data\sensitivity_matrix.npy',sensitivity)