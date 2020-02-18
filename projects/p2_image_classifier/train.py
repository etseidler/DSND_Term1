import argparse
from torch import cuda
from model import build_model_from_pretrained, save_checkpoint, train, ALLOWED_ARCHS
from utils import prep_data
from workspace_utils import active_session


def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint')
    parser.add_argument('data_dir', metavar='path/to/dataset', type=str, nargs=1, help='path to a data directory')
    parser.add_argument('--save_dir', metavar='path/to/save_dir', type=str, nargs='?', help='path to a directory in which to save a checkpoint')
    parser.add_argument('--learning_rate', metavar='learning rate', type=float, nargs='?', default=0.002, help='learning rate value for model training')
    parser.add_argument('--hidden_units', metavar='hidden units', type=int, nargs='?', default=512, help='number of hidden units for model classifier')
    parser.add_argument('--epochs', metavar='epochs', type=int, nargs='?', default=5, help='number of epochs for model training')
    parser.add_argument('--arch', metavar='model name', type=str, nargs='?', default='densenet161', help='name of transfer model (e.g., resnet18 or densenet161)')
    parser.add_argument('--gpu', action='store_true', help='use GPU for model training (recommended)')
    args = parser.parse_args()
    data_dir = args.data_dir[0]
    save_dir = args.save_dir
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    model_name = args.arch
    use_gpu = args.gpu
    print('Using the following hyperparameters for training')
    print(f'  Learning rate: {learning_rate}')
    print(f'  Hidden units: {hidden_units}')
    print(f'  Epochs: {epochs}')
    print(f'Transfer model name: {model_name}')
    if use_gpu and not cuda.is_available():
        print('Error: GPU not available. Try again without the --gpu flag')
        exit(1)
    if use_gpu:
        print('Training on GPU...')
    else:
        print('Training on CPU...')
        print('Warning: training on CPU could take a LONG time. Consider using --gpu flag.')
    print('')
    
    if model_name not in ALLOWED_ARCHS:
        print(f'Error: Model architecture {model_name} is not currently supported.')
        print('Please try one of the following:')
        for a in ALLOWED_ARCHS:
            print(f'  {a}')
        exit(1)
    
    dataloaders, image_datasets = prep_data(data_dir)
    
    model = build_model_from_pretrained(model_name, hidden_units, image_datasets['test'].class_to_idx)
#     print(model.classifier)

    with active_session():
        trained, optimizer = train(model, dataloaders, learning_rate, epochs, use_gpu)
    
    if save_dir:
        save_checkpoint(trained, epochs, optimizer, model_name, learning_rate, save_dir)

    
if __name__ == '__main__':
    main()
