import argparse
import json
from math import trunc
from utils import get_torched_image
from model import build_model_from_state_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description='Predict type of flower using a pretrained model and custom classifier')
    parser.add_argument('image_path', metavar='path/to/image', type=str, nargs=1, help='path to an image (of a flower)')
    parser.add_argument('checkpoint_path', metavar='path/to/checkpoint', type=str, nargs=1, help='path to a checkpoint of a trained model')
    parser.add_argument('--top_k', metavar='top_k', type=int, nargs='?', help='return this number of top k classes (not names)')
    parser.add_argument('--gpu', action='store_true', help='use GPU when doing prediction')

    args = parser.parse_args()
    image_path = args.image_path[0]
    checkpoint_path = args.checkpoint_path[0]
    top_k = args.top_k
    use_gpu = args.gpu
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    img = get_torched_image(image_path)

    model, _ = build_model_from_state_checkpoint(checkpoint_path)

    name, prob, classes = predict(img, model, cat_to_name, top_k or 1, use_gpu)

    if top_k:
        print(f'Top {top_k} most likely classes (not names): {classes}')
    else:
        print(f'Flower name: {name.capitalize()}')
        print(f'Probability: {trunc(100*prob)}%')


if __name__ == '__main__':
    main()
