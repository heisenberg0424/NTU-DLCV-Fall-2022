import argparse
import cv2
import glob
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from tokenizers import Tokenizer
from typing import List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from mypredict import load_checkpoint

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
# create caption and mask
def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template.cuda(), mask_template.cuda()

# open images and convert to RGB
def get_images(root, transform = None):
	images = []
	fns = []
	filenames = glob.glob(os.path.join(root, '*.jpg'))
	for fn in filenames:
		image = Image.open(fn).convert('RGB')
		image = transform(image)
		images.append(image)
		fns.append(fn)

	return images, fns

# test the model
# ref: https://github.com/saahiluppal/catr/blob/fac82f9b4004b1dd39ccf89760b758ad19a2dbee/models/transformer.py
def test(model, caption, max_position_embeddings, cap_mask, image, device):
	# fig num
	fig_num = 1

	# set the model to the evaluation mode
	model.eval()

	# add a additional dimension to the image
	image_shape = image.shape
	print(image_shape)
	image = image.unsqueeze(0)

	atten_map = []

	with torch.no_grad():
		for i in range(max_position_embeddings - 1):

			# put the data to the GPU
			image, caption, cap_mask = image.to(device), caption.to(device), cap_mask.to(device)

			if not isinstance(image, NestedTensor):
				image = nested_tensor_from_tensor_list(image)

			# backbone
			features, pos = model.backbone(image)
			src, mask = features[-1].decompose()
			assert mask is not None

			# transformer
			src = model.input_proj(src)
			bs, c, h, w = src.shape
			src = src.flatten(2).permute(2, 0, 1)
			pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
			mask = mask.flatten(1)

			tgt = model.transformer.embeddings(caption).permute(1, 0, 2)
			query_embed = model.transformer.embeddings.position_embeddings.weight.unsqueeze(1)
			query_embed = query_embed.repeat(1, bs, 1)

			# transformer encoder
			memory = src

			for j, layer in enumerate(model.transformer.encoder.layers):
				# transformer encoder layer
				if layer.normalize_before:
					# forward pre
					src2 = layer.norm1(memory)
					q = k = layer.with_pos_embed(src2, pos_embed)
					src2 = layer.self_attn(q, k, value = src2, attn_mask = None, key_padding_mask = mask) #(output, weights)
					src2 = src2[0]
					memory = memory + layer.dropout1(src2)
					src2 = layer.norm2(memory)
					src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src2))))
					memory = memory + layer.dropout2(src2)
				else:
					# forward post
					q = k = layer.with_pos_embed(src2, pos_embed)
					src2 = layer.self_attn(q, k, value = src2, attn_mask = None, key_padding_mask = mask) #(output, weights)
					src2 = src2[0]
					memory = memory + layer.dropout1(src2)
					src2 = layer.norm1(memory)
					src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src2))))
					memory = memory + layer.dropout2(src2)
					memory = layer.norm2(memory)

			if model.transformer.encoder.norm is not None:
				memory = model.transformer.encoder.norm(memory)

			# transformer decoder
			hs = tgt
			intermediate = []

			for j, layer in enumerate(model.transformer.decoder.layers):
				
				# transformer decoder layer
				if layer.normalize_before:
					tgt2 = layer.norm1(hs)
					q = k = layer.with_pos_embed(tgt2, query_embed)
					tgt2 = layer.self_attn( # self-attention with caption
								q, k, value = tgt2, 
								attn_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device),
								key_padding_mask = cap_mask)[0]
					hs = hs + layer.dropout1(tgt2)
					tgt2 = layer.norm2(hs)
					tgt2 = layer.multihead_attn( # cross-attention
								query = layer.with_pos_embed(tgt2, query_embed),
								key = layer.with_pos_embed(memory, pos_embed),
								value = memory,
								attn_mask = None,
								key_padding_mask = mask,
							)
					if j == len(model.transformer.decoder.layers) - 1:
						scores = torch.reshape(tgt2[1][0, i], (int(tgt2[1].shape[2]/19), 19)).cpu().numpy()
						scores = cv2.resize(scores, [image_shape[2], image_shape[1]], interpolation = cv2.INTER_CUBIC)
						atten_map.append(scores) # tgt2: (batch, max_position_embedding, patches)
						
					tgt2 = tgt2[0]
					hs = hs + layer.dropout2(tgt2)
					tgt2 = layer.norm3(hs)
					tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
					hs = hs + layer.dropout3(tgt2)

				else:
					q = k = layer.with_pos_embed(tgt2, query_embed)
					tgt2 = layer.self_attn( # self-attention with caption
								q, k, value = tgt2, 
								attn_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device),
								key_padding_mask = cap_mask)[0]					
					hs = hs + layer.dropout1(tgt2)
					tgt2 = layer.norm1(hs)
					tgt2 = layer.multihead_attn( # cross-attention
								query = layer.with_pos_embed(tgt2, query_embed),
								key = layer.with_pos_embed(memory, pos_embed),
								value = memory,
								attn_mask = None,
								key_padding_mask = mask,
							)
					if j == len(model.transformer.decoder.layers) - 1:
						scores = torch.reshape(tgt2[1][0, i], (int(tgt2[1].shape[2]/19), 19)).cpu().numpy()
						scores = cv2.resize(scores, [image_shape[1], image_shape[2]], interpolation = cv2.INTER_CUBIC)
						atten_map.append(scores) # tgt2: (batch, max_position_embedding, patches)

					tgt2 = tgt2[0]
					hs = hs + layer.dropout2(tgt2)
					tgt2 = layer.norm2(hs)
					tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
					hs = hs + layer.dropout3(tgt2)
					hs = layer.norm3(hs)
				
				if model.transformer.decoder.return_intermediate:
					intermediate.append(model.transformer.decoder.norm(hs))

			if model.transformer.decoder.norm is not None:
				hs = model.transformer.decoder.norm(hs)
				if model.transformer.decoder.return_intermediate:
					intermediate.pop()
					intermediate.append(hs)

			if model.transformer.decoder.return_intermediate:
				hs = torch.stack(intermediate)

			# MLP
			predictions = model.mlp(hs.permute(1, 0, 2))

			predictions = predictions[:, i, :]
			predicted_id = torch.argmax(predictions, axis=-1)

			if predicted_id[0] == 3:
				return caption, atten_map

			caption[:, i+1] = predicted_id[0]
			cap_mask[:, i+1] = False
			

	return caption, atten_map

# change the long side dimension to MAX_DIM
def under_max(image):
	
	MAX_DIM = 299

	if image.mode != 'RGB':
		image = image.convert("RGB")

	shape = np.array(image.size, dtype = float)
	long_dim = max(shape)
	scale = MAX_DIM / long_dim

	new_shape = (shape * scale).astype(int)
	image = image.resize(new_shape)

	return image

if __name__ == '__main__':

	# max position embeddings
	max_position_embeddings = 128

	# parse_args
	parser = argparse.ArgumentParser()
	parser.add_argument('--testset_path', help = 'input the path of the testset', default = "../../hw3_data/p3_data/images")
	parser.add_argument('--output_path', help = 'input the path of the output dir', default = './')
	args = parser.parse_args()

	"""check if GPU is availabl"""
	device = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device)
	print('Device: ', device)

	# load catr model
	model = load_checkpoint('hw3_2.pth')
	model.cuda()
	tokenizer = Tokenizer.from_file("../../hw3_data/caption_tokenizer.json")


	# transform
	transform = transforms.Compose([
		transforms.Lambda(under_max),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	# get testing images]
	images, fns = get_images(root = args.testset_path, transform = transform)

	print(f'# of images: {len(images)}')

	# get the caption from the output of model
	for i in range(len(images)):
		# get caption and mask, the caption contains <sos>
		caption, cap_mask = create_caption_and_mask(0, max_position_embeddings)
		
		output, atten_map = test(model, caption, max_position_embeddings, cap_mask, images[i], device)
		result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
		res = result.split(' ')
		print(result.capitalize())

		image_np = np.array((images[i] + 1) * 255 / 2, dtype = int)
		
		plt.figure(i+1)
		# plot the start part
		plt.subplot(int((len(atten_map) - 1) / 5 + 1), 5, 1)
		plt.imshow(image_np.transpose(1, 2, 0))
		plt.title('<start>', fontsize = 8)
		plt.axis('off')
		plt.subplots_adjust(top = 0.92, bottom = 0.08, left = 0.03, right = 0.97, hspace = 0, wspace = 0.04)

		# plot the caption part
		for j in range(len(atten_map) - 1):
			plt.subplot(int((len(atten_map) - 1) / 5 + 1), 5, j+2)
			plt.imshow(image_np.transpose(1, 2, 0))
			plt.imshow(atten_map[j], cmap = 'jet', alpha = 0.6, interpolation = 'gaussian')
			
			if j < len(atten_map) - 2:
				#plt.title(tokenizer.decode(output[0,j+1], skip_special_tokens=True), fontsize = 8)
				plt.title(res[j], fontsize = 8)
			# replace the '.'
			else:
				plt.title('<end>', fontsize = 8)

			plt.axis('off')

		print(fns[i])
		plt.savefig(os.path.join(args.output_path, fns[i].split('/')[-1].split('.')[0] + '.png'))
		#plt.show()