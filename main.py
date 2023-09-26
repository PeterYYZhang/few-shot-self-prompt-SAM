import cv2
import numpy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import os
import random
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import argparse
from utils.utils import *
import time


def get_embedding(img, predictor):
    predictor.set_image(img)
    img_emb = predictor.get_image_embedding()
    return img_emb


def train(args, predictor):
    data_path = args.data_path
    assert os.path.exists(data_path), 'data path does not exist!'

    num_image = args.k

    fnames = os.listdir(os.path.join(data_path, 'images'))
    # get 20 random indices from fnames
    random.shuffle(fnames)
    fnames = fnames[:num_image]
    image_embeddings = []
    labels = []
    
    # get the image embeddings
    print('Start training...')
    t1 = time.time()
    i = 0 
    for fname in tqdm(fnames):
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY) # threshold the mask to 0 and 1
        downsampled_mask = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
         
        img_emb = get_embedding(image, predictor)
        img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
        image_embeddings.append(img_emb)

        labels.append(downsampled_mask.flatten())
        
        i += 1
        if i > num_image: break
    t2 = time.time()
    print("Time used: {}m {}s".format((t2 - t1) // 60, (t2 - t1) % 60))
    image_embeddings_cat = np.concatenate(image_embeddings)
    labels = np.concatenate(labels)

    # Create a linear regression model and fit it to the training data
    model = LogisticRegression(max_iter=1000) 
    model.fit(image_embeddings_cat, labels)
    
    return model

def test_visualize(args, model, predictor):
    data_path = args.data_path
        
    num_image = args.k
    fnames = os.listdir(os.path.join(data_path, 'images'))
    random.shuffle(fnames)
    fnames = fnames[num_image:]
    num_visualize = args.visualize_num
    
    dice_linear = []
    dice1 = []
    dice2 = []
    dice3 = []
    i = 0

    for fname in tqdm(fnames[:num_visualize]):
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        H, W, _ = image.shape
        
        # get the image embedding and flatten it
        img_emb = get_embedding(image, predictor)
        img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
        
        # get the mask predicted by the linear classifier
        y_pred = model.predict(img_emb)
        y_pred = y_pred.reshape((64, 64))
        # mask predicted by the linear classifier
        mask_pred_l = cv2.resize(y_pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # use distance transform to find a point inside the mask
        fg_point = get_max_dist_point(mask_pred_l)
        # Define the kernel for dilation
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask_pred_l, kernel, iterations=3)
        mask_pred_l = cv2.dilate(eroded_mask, kernel, iterations=5)
        
        # set the image to sam
        predictor.set_image(image)
        
        # prompt the sam with the point
        input_point = np.array([[fg_point[0], fg_point[1]]])
        input_label = np.array([1])
        masks_pred_sam_prompted1, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )
        
        # prompt the sam with the bounding box
        y_indices, x_indices = np.where(mask_pred_l > 0)
        if np.all(mask_pred_l == 0):
            bbox = np.array([0, 0, H, W])
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = mask_pred_l.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            bbox = np.array([x_min, y_min, x_max, y_max])
        masks_pred_sam_prompted2, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,)
            
        # prompt the sam with both the point and bounding box
        masks_pred_sam_prompted3, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=bbox[None, :],
            multimask_output=False,)
        
        dice_l = dice_coef(mask, mask_pred_l)
        dice_p = dice_coef(mask, masks_pred_sam_prompted1[0])
        dice_b = dice_coef(mask, masks_pred_sam_prompted2[0])
        dice_i = dice_coef(mask, masks_pred_sam_prompted3[0])
        dice_linear.append(dice_l)
        dice1.append(dice_p)
        dice2.append(dice_b)
        dice3.append(dice_i)

        # plot the results
        fig, ax = plt.subplots(1, 5, figsize=(15, 10))
        ax[0].set_title('Ground Truth')
        ax[0].imshow(mask)
        ax[1].set_title('Linear + e&d')
        ax[1].plot(fg_point[0], fg_point[1], 'r.')
        ax[1].imshow(mask_pred_l)
        ax[2].set_title('Point')
        ax[2].plot(fg_point[0], fg_point[1], 'r.')
        ax[2].imshow(masks_pred_sam_prompted1[0]) 
        ax[3].set_title('Box')
        show_box(bbox, ax[3])
        ax[3].imshow(masks_pred_sam_prompted2[0])
        ax[4].set_title('Point + Box')
        ax[4].plot(fg_point[0], fg_point[1], 'r.')
        show_box(bbox, ax[4])
        ax[4].imshow(masks_pred_sam_prompted3[0])
        [axi.set_axis_off() for axi in ax.ravel()]
        
        
        if os.path.exists(args.save_path) == False:
            os.mkdir(args.save_path)
        plt.savefig(os.path.join(args.save_path, fname.split('.')[0]+str(i)))
    
    mdice0 = round(sum(dice_linear)/float(len(dice_linear)), 5)
    mdice1 = round(sum(dice1)/float(len(dice1)), 5)
    mdice2 = round(sum(dice2)/float(len(dice2)), 5)
    mdice3 = round(sum(dice3)/float(len(dice3)), 5)
    
    print('For the first {} images: '.format(num_visualize))
    print('mdice(linear classifier: )', mdice0)
    print('mDice(point prompts): ', mdice1)
    print('mDice(bbox prompts): ', mdice2)
    print('mDice(points and boxes): ', mdice3)

        
        
def test(args, predictor):
    data_path = args.data_path
    images = []
    masks = []
    fnames = os.listdir(os.path.join(data_path, 'images'))
    print(f'loading images from {data_path}...')
    for fname in tqdm(fnames):
        # read data
        image = cv2.imread(os.path.join(data_path, 'images', fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(data_path, 'masks', fname))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        images.append(image)
        masks.append(mask)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, text_index in kf.split(images):
        train_images = [images[i] for i in train_index]
        train_masks = [masks[i] for i in train_index]
        test_images = [images[i] for i in text_index]
        test_masks = [masks[i] for i in text_index]
        
        # train the linear classifier
        k = args.k
        random_indices = random.sample(range(len(train_images)), k)
        image_embeddings = []
        labels = []
        for idx in random_indices:
            image = train_images[idx]
            mask = train_masks[idx]
            downsampled_mask = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)

            img_emb = get_embedding(image)
            img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)
            image_embeddings.append(img_emb)
            labels.append(downsampled_mask.flatten())
                
        image_embeddings_cat = numpy.concatenate(image_embeddings)
        labels = numpy.concatenate(labels)

        model = LogisticRegression(max_iter=1000) # how to set parameters?? C, max_iter, verbose, solver
        model.fit(image_embeddings_cat, labels)

        # test
        dice_linear=[]
        dice1=[]
        dice2=[]
        dice3=[]
        for idx in range(len(test_images)):
            image = test_images[idx]
            mask = test_masks[idx]
            H, W, _ = image.shape

            img_emb = get_embedding(image)
            img_emb = img_emb.cpu().numpy().transpose((2, 3, 1, 0)).reshape((64, 64, 256)).reshape(-1, 256)

            # ger the mask predicted by the linear classifier
            y_pred = model.predict(img_emb)
            y_pred = y_pred.reshape((64,64))
            mask_pred_l = cv2.resize(y_pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # use distance transform to find a point inside the mask
            fg_point = get_max_dist_point(mask_pred_l)

            # Define the kernel for dilation
            kernel = np.ones((5, 5), np.uint8)

            eroded_mask = cv2.erode(mask_pred_l, kernel, iterations=3)
            mask_pred_l = cv2.dilate(eroded_mask, kernel, iterations=5)

            # set the image to sam
            predictor.set_image(image)

            # prompt sam with the point
            input_point = np.array([[fg_point[0], fg_point[1]]])
            input_label = np.array([1])
            masks_pred_sam_prompted1, _, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=None,
            multimask_output=False,)

            # prompt sam with the bbox
            y_indices, x_indices = np.where(mask_pred_l > 0)
            if np.all(mask_pred_l==0):
                bbox = np.array([0 ,0, H, W])
            else:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                H, W = mask_pred_l.shape
                x_min = max(0, x_min - np.random.randint(0, 20))
                x_max = min(W, x_max + np.random.randint(0, 20))
                y_min = max(0, y_min - np.random.randint(0, 20))
                y_max = min(H, y_max + np.random.randint(0, 20))
                bbox = np.array([x_min, y_min, x_max, y_max])
                masks_pred_sam_prompted2, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,)

                masks_pred_sam_prompted3, _, _,= predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=bbox[None, :],
                multimask_output=False,)

                dice_l = dice_coef(mask, mask_pred_l)
                dice_p = dice_coef(mask, masks_pred_sam_prompted1[0])
                dice_b = dice_coef(mask, masks_pred_sam_prompted2[0])
                dice_c = dice_coef(mask, masks_pred_sam_prompted3[0])
                dice_linear.append(dice_l)
                dice1.append(dice_p)
                dice2.append(dice_b)
                dice3.append(dice_c)
                
        mdice0 = round(sum(dice_linear)/float(len(dice_linear)), 5)
        mdice1 = round(sum(dice1)/float(len(dice1)), 5)
        mdice2 = round(sum(dice2)/float(len(dice2)), 5)
        mdice3 = round(sum(dice3)/float(len(dice3)), 5)

        print('mdice(linear classifier: )', mdice0)
        print('mDice(point prompts): ', mdice1)
        print('mDice(bbox prompts): ', mdice2)
        print('mDice(points and boxes): ', mdice3)
        print('\n')

    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--k', type=int, default=10, help='number of pics')
    parser.add_argument('--data_path', type=str, default='./data/Kvasir-SEG', help='path to train data')
    parser.add_argument('--model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/sam_vit_b_01ec64.pth', help='SAM checkpoint')
    parser.add_argument('--visualize', type=bool, default=True, help='visualize the results')
    parser.add_argument('--save_path', type=str, default='./results', help='path to save the results')
    parser.add_argument('--visualize_num', type=int, default=30, help='number of pics to visualize')
    args = parser.parse_args()

    # set random seed
    random.seed(42)
    
    # register the SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    global predictor
    predictor = SamPredictor(sam)
    print('SAM model loaded!', '\n')
    
    if args.visualize:
        model = train(args, predictor)
        test_visualize(args, model, predictor)
    else:
        test(args, predictor)


if __name__ == '__main__':
    main()