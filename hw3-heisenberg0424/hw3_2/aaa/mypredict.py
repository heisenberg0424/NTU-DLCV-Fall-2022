import os
import glob
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys
from models import caption
from configuration import Config
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print("device:", device)

config = Config()

class CATR_dataset(Dataset):
    def __init__(self, root, transform = None):
        self.root = root
        self.images = None
        self.labels = None
        self.filename = []
        self.transform = transform
        
        filenames = glob.glob(os.path.join(root, '*.jpg'))
        for fn in filenames:
            self.filename.append(fn)

        self.len = len(self.filename)

    def __getitem__(self, idx):
        image_fn = self.filename[idx]
        image = Image.open(image_fn).convert('RGB')
        image_fn_split = image_fn.split('/')[-1]

        if self.transform is not None:
            image_tfm = self.transform(image)

        return image_tfm, image_fn_split

    def __len__(self):
        return self.len


def evaluate(model, caption, image):
    start_token = 0
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 3:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption

def predict(model, testset, caption, tokenizer,outputfile):
    caption_result = dict()

    for i in tqdm(testset):
        image, filename = i
        filename, _ = filename.split('.')
        image = image.unsqueeze(0).to(device)
        output_caption = evaluate(model, caption, image)
        result = tokenizer.decode(output_caption[0].tolist(), skip_special_tokens=True)
        # print(result.capitalize())

        caption_result[filename] = result.capitalize()
        json_object = json.dumps(caption_result, indent=4)
        with open(outputfile, "w") as outfile:
            outfile.write(json_object)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template.to(device), mask_template.to(device)

def under_max(image):
    MAX_DIM = 224

    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float64)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image

def load_checkpoint(checkpoint_path):
    print("Loading Checkpoint...")
    model, _ = caption.build_model(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    return model

def main():

    model = load_checkpoint('hw3_2.pth')
    model.to(device)
    val_transform = transforms.Compose([
    transforms.Lambda(under_max),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = CATR_dataset(sys.argv[1], val_transform)
    

    tokenizer = Tokenizer.from_file("caption_tokenizer.json")

    predict(model, testset, caption, tokenizer,sys.argv[2])

if __name__ == '__main__':
    main()
