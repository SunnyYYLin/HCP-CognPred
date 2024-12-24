import numpy as np

def calculate_column_means(input_file, output_file):
    data = np.load(input_file)
    column_means = np.mean(data, axis=0)
    np.savetxt(output_file, column_means.reshape(1, -1), fmt='%.6f')

if __name__ == "__main__":
    input_file = r'data\rsfc_Yeo400_753_GSR.npy'  
    output_file = r'data\mean_GSR.npy'  
    calculate_column_means(input_file, output_file)