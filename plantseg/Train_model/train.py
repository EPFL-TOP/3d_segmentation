import torch
from pytorch3dunet.unet3d.trainer import create_trainer
import yaml
import logging

'''
Important information:
1. pytorch3dunet expects the data to be in HDF5 format.
2. The training and validation data should have a raw and label group.
3. When choosing the patch size and num workers, consider the GPU memory and the size of your dataset.
4. If you want to retrain a model, use the pretrained in the trainer config. However, make sure the model architecture is compatible with the new data.
You can use this :

        # Load your saved state_dict
        model_state = torch.load('/path/to/best_checkpoint.pytorch', map_location='cpu')

        # Wrap it into the expected format
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': None,  # Can be real if available, or leave None
            'num_epochs': 0,
            'num_iterations': 0,
            'best_eval_score': 0.0
        }

        # Save the full checkpoint
        torch.save(checkpoint, '/path/to/full_checkpoint.pytorch')
5. if you want to see what is inside the model, you can use the following code:
        checkpoint = torch.load("/mnt/e/PROJECTS-01/Adrian/Bachelor_project/Train/model/last_checkpoint03.pytorch", map_location="cpu")
        # To see what's inside:
        print(checkpoint.keys())
        print("Saved at epoch:", checkpoint['num_epochs'])
        print("best_eval_score:", checkpoint['best_eval_score'])
        state_dict = checkpoint['model_state_dict']
        i=0
        for name, param in state_dict.items():
            print(name, param.shape)
            if i == 5:
                break
6. if the training is interrupted, you can resume it by using the `resume` parameter in the trainer config.
'''

print('GPU active : ',torch.cuda.is_available())

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Use logging to track the training process, can be useful for debugging, monitoring and analysis
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Redirect logs to a file
logging.basicConfig(
    filename='training_log5.txt',
    level=logging.INFO,  # Or DEBUG
    format='%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s',
)

Zebra=create_trainer(config)

Zebra.fit()

