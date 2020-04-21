import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # training settings     
    parser.add_argument('--beam', type=int, default=0,
                    help='whether use beam search in validation')    
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--num_gpu', type=int, default=1,
                    help='num of gpus')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='gpu_id')  
    parser.add_argument('--threshold', type=float, default=1.05,
                    help='threshold of saving checkpoints')       
    parser.add_argument('--selected_num', type=int, default=15,
                    help='num of selected attributes')    
    parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
    parser.add_argument('--cached_tokens', type=str, default='coco-train-new-idxs',
                    help='Cached token file for calculating cider score during self critical training.')
    parser.add_argument('--self_critical_after', type=int, default=100,
                    help='After what epoch do we start SCST? (-1 = disable; never finetune, 0 = finetune from start)')      
    parser.add_argument('--train_split', type=str, default='train',
                        help='which split used to train')    
    # Data directories
    parser.add_argument('--input_json', type=str, default='data/cocotalk_attr.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='data/cocobu_fc_36',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='data/cocobu_att_36',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='data/cocobu_box_36',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_label_h5', type=str, default='data/cocotalk_attr_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
        
    parser.add_argument('--input_attr_label_dir', type=str, default='data/coco_mask_att',
                    help='path to the directory containing the labels of atttributes')    
    parser.add_argument('--input_subsequent_label_dir', type=str,default='data/coco_tran_label',
                        help='path to the directory containing labels of subsequent atttributes')
    


    # Model settings
    parser.add_argument('--caption_model', type=str, default="lstm_MAD_SAP",
                    help='only support MAD+SAP now')
    parser.add_argument('--id', type=str, default='MAD_SAP',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')    
    parser.add_argument('--rnn_size', type=int, default=1000,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
  
    parser.add_argument('--input_encoding_size', type=int, default=1000,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')



    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=1000,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=240,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                    help='clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')

    parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='learning rate')

    parser.add_argument('--learning_rate_decay_start', type=int, default=0, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=5, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                        help='every how many iterations thereafter to drop LR?(in epoch)')

    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay used for adam')    
    parser.add_argument('--accumulate_number', type=int, default=1,
                        help='how many times it should accumulate the gradients, the truth batch_size=accumulate_number*batch_size')

    parser.add_argument('--scheduled_sampling_start', type=int, default=0, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=-1,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=500,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=10,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       



    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
  
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"


    return args
