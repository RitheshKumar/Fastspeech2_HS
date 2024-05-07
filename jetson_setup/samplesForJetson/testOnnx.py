import torch
import pandas as pd
import onnxruntime as ort
# import sounddevice as sd
import soundfile as sf
import numpy as np
import time


ort_sess = ort.InferenceSession('./hifigan_mac.onnx',providers=[("CUDAExecutionProvider",{"cudnn_conv_algo_search":"DEFAULT"})])

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print("Device is",device)

#Get mels and convert to a model readable format
df = pd.read_csv("./Hindi_Mels_1")
data= df.values

tensor = torch.from_numpy(data)
my_tensor=tensor[:, :-1]
my_tensor=my_tensor.T


# Define the dimensions for the new 3D tensor
new_shape = (1, my_tensor.size(0), my_tensor.size(1))

# Reshape the tensor to 3D
my_3d_tensor = my_tensor.view(new_shape)

x = my_3d_tensor.float()
x = x.to('cpu') #copy tensor to cpu first. run command copies to cuda

start = time.time()

outputs = ort_sess.run(None, {'input': x.numpy()})
data = outputs[0][0][0]

print("Model runtime is",time.time()-start)

sample_rate = 48000

# Normalize the float array to be between -1 and 1
data /= np.max(np.abs(data))

# # Play the audio
# sd.play(data, sample_rate)
# sd.wait()  # Wait until the audio is done playing

# Save the audio data to a WAV file
output_filename = './output_CUDA.wav' 
sf.write(output_filename, data, sample_rate)
