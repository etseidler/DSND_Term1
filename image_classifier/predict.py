import argparse
import json
from math import trunc
from torch import cuda
from utils import get_torched_image
from model import build_model_from_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description='Predict type of flower using a pretrained model and custom classifier')
    parser.add_argument('image_path', metavar='path/to/image', type=str, nargs=1, help='path to an image (of a flower)')
    parser.add_argument('checkpoint_path', metavar='path/to/checkpoint', type=str, nargs=1, help='path to a checkpoint of a trained model')
    parser.add_argument('--top_k', metavar='top_k', type=int, nargs='?', help='return this number of top k classes')
    parser.add_argument('--category_names', metavar='path/to/mapping', type=str, nargs='?', help='path to file that maps classes to flower names so results will show names (not classes)')
    parser.add_argument('--gpu', action='store_true', help='use GPU when doing prediction')
    args = parser.parse_args()
    image_path = args.image_path[0]
    checkpoint_path = args.checkpoint_path[0]
    top_k = args.top_k
    use_gpu = args.gpu
    cat_to_names = args.category_names
    if use_gpu and not cuda.is_available():
        print('Error: GPU not available. Try again without the --gpu flag')
        exit(1)

    img = get_torched_image(image_path)

    model, _ = build_model_from_checkpoint(checkpoint_path)

    classes, probs = predict(img, model, top_k or 1, use_gpu)
    probs = [str(trunc(100*p)) + '%' for p in probs]
    if cat_to_names:
        with open(cat_to_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c].capitalize() for c in classes]
    
    if top_k:
        classes_or_names = 'names' if cat_to_names else 'classes'
        joiner = ', '
        print(f'Top {top_k} most likely {classes_or_names}: {joiner.join(classes)}')
        print(f'Associated probabilities: {joiner.join(probs)}')
    else:
        class_or_name = 'name' if cat_to_names else 'class'
        print(f'Flower {class_or_name}: {classes[0]}')
        print(f'Probability: {probs[0]}')


if __name__ == '__main__':
    main()
