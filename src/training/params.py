from typing import Tuple
import argparse
import ast


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split("=")
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(
                    value
                )  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train-data",
        type=str,
        default="/scratch/project_465000727/datasets/img2dataset/mscoco1024_centercrop/{00000..00577}.tar",
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        ),
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=100,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="webdataset",
        help="Which type of dataset to process.",
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection.",
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use.",
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.",
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.",
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=7, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown",
        type=int,
        default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards.",
    )
    parser.add_argument("--lr", type=float, default=5.0e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1.0e-6, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    # parser.add_argument(
    #     "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    # )
    parser.add_argument(
        "--warmup", type=float, default=0.1, help="Fraction of steps in warmup mode."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end",
        type=float,
        default=0.0,
        help="End learning rate for cooldown schedule. Default: 0",
    )
    parser.add_argument(
        "--lr-cooldown-power",
        type=float,
        default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=20, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=[
            "amp",
            "amp_bf16",
            "amp_bfloat16",
            "bf16",
            "fp16",
            "pure_bf16",
            "pure_fp16",
            "fp32",
        ],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ours_dinov2_ViT-B-14",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action="store_true",
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action="store_true",
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs="+",
        default=None,
        metavar="MEAN",
        help="Override default image mean value of dataset",
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs="+",
        default=None,
        metavar="STD",
        help="Override default image std deviation of of dataset",
    )
    parser.add_argument("--aug-cfg", nargs="*", default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)",
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather",
    )
    parser.add_argument(
        "--force-image-size",
        type=int,
        nargs="+",
        default=None,
        help="Override default image size",
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action="store_true",
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action="store_true",
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action="store_true",
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default="wandb",
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']",
    )
    parser.add_argument(
        "--wandb-notes", default="", type=str, help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="open-clip",
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Option to disable wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there.",
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Default random seed.")
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action="store_true",
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action="store_true",
        help="Freeze BatchNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=2,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa.",
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa.",
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one.",
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help="Which model arch to distill from, if any.",
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help="Which pre-trained weights to distill from, if any.",
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help="Replace the network linear layers from the bitsandbytes library. "
        "Allows int8 training/inference, etc.",
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help="Use SigLip (sigmoid) loss.",
    )

    ######### ADDED ARGS #########
    parser.add_argument("--ours", default=True, help="Use our loss.")
    parser.add_argument(
        "--lambda_context",
        default=0.1,
        type=float,
        help="Weight given to the text representation.",
    )
    parser.add_argument(
        "--concepts_path",
        default="/scratch/project_465000727/datasets/imagenet12k/imagenet12k_concept_clean.txt",
        type=str,
        help="Path to concepts as a text file (one line per concept).",
    )
    parser.add_argument(  # TODO: remove
        "--concepts_embeddings_path",
        default="/scratch/project_465000727/open_clip_metadata/vitb16_laion2b_s34b_b88k/coco_names_embedded.pth",
        type=str,
        help="Path to pre-computed concepts.",
    )
    parser.add_argument(
        "--num_heads_context",
        default=1,
        type=int,
        help="Number of heads in the cross-modality attention layer.",
    )
    parser.add_argument(
        "--global_crops_scale",
        default=(0.25, 1.0),
        type=Tuple[float, float],
        help="Number of heads in the cross-modality attention layer.",
    )
    parser.add_argument(
        "--patch_size", default=14, type=int, help="Patch-size of the ViT."
    )  # TODO: remove as it can/should be inferred
    parser.add_argument("--n_tokens", default=1, type=int, help="Number of centroids.")
    parser.add_argument(
        "--pos_alpha",
        default=0.0,
        type=int,
        help="Weight given to spatial consistency in the clustering algorithm.",
    )
    # parser.add_argument(
    #     "--beta_ema",
    #     default=1.0,
    #     type=float,
    # )
    parser.add_argument(
        "--coco-val",
        type=str,
        default="/scratch/project_465000727/datasets/coco",
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--visual_concepts_init_path",
        type=str,
        default="/scratch/project_465000727/repos/Contextual-CLIP/log_cclip/000003_new_concepts_imagenet12k_reg_timm/visual_concepts_imagenet12k_dinov2-base-14-reg.pth",
        help="Path to vision concepts.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "init", "init_kmeans", "init_imagenet12k"],
        help="Mode for main.py script.",
    )
    parser.add_argument(
        "--image_features_type",
        type=str,
        default="cls",
        choices=["cls", "pacl"],
        help="Type of image features to use.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="main",
        choices=["cls", "main"],
        help="Type of image features to use.",
    )
    parser.add_argument(
        "--max_nb_img_kmeans",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--nb_iter_kmeans_init",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--model_origin",
        type=str,
        default="timm",
        choices=["timm", "torchhub"],
        help="Which model provider to use.",
    )
    parser.add_argument(
        "--match_type",
        type=str,
        default="textual",
        choices=["textual", "visual"],
        help="Which matchings to use.",
    )
    parser.add_argument(
        "--coco_things_classes_path",
        type=str,
        default="/scratch/project_465000727/repos/Contextual-CLIP/src/training/class_names/coco_things_classes.json",
        help="Path to coco_things_classes.",
    )
    parser.add_argument(
        "--n_dense_cls",
        type=int,
        default=0,
        help="Number of dense [CLS] tokens in the visual and text models.",
    )
    parser.add_argument(
        "--visual_norm",
        type=bool,
        default=False,
        help="Use the last norm of the visual model.",
    )
    parser.add_argument(
        "--n_decoder_layers",
        type=int,
        default=4,
        help="Number of decoder layers.",
    )
    parser.add_argument(
        "--n_decoder_heads",
        type=int,
        default=4,
        help="Number of heads in the decoder.",
    )
    parser.add_argument(
        "--ratio_to_keep_dense",
        type=float,
        default=1.0,
        help="Ratio of dense matchings to keep.",
    )
    parser.add_argument(
        "--decoder_type",
        type=str,
        default="TransformerDecoder",
        choices=["TransformerDecoder", "shallow"],
        help="Which decoder to use.",
    )
    parser.add_argument(
        "--decoder_softmax_dim",
        type=str,
        default="patch",
        choices=["patch", "n_dense_cls"],
        help="Dimension for softmax in decoder.",
    )
    parser.add_argument(
        "--nouns_path",
        type=str,
        default="/scratch/project_465000727/thomas_repos/Contextual-CLIP_dev/src/training/class_names/v3det_classes.txt",
        help="Path to file containing the nouns.",
    )
    parser.add_argument(
        "--mask_type_text",
        type=str,
        default="causal",
        choices=["causal", "None"],
        help="Which mask type to use in text transformer.",
    )
    parser.add_argument(
        "--indices_type",
        type=str,
        default="np",
        choices=["np", "concept"],
        help="Use concepts or noun phrases.",
    )
    parser.add_argument(
        "--mask_type_loss",
        type=str,
        default="None",
        choices=["same_word", "None"],
        help="Which mask type to use in loss.",
    )
    parser.add_argument(
        "--temp_decoder",
        type=float,
        default=1.0,
        help="Temperature for decoder contrastive loss.",
    )
    parser.add_argument(
        "--denoiser_type",
        type=str,
        default=None,
        choices=[None, "transformer"],
        help="Which denoiser type to use in ViT.",
    )
    parser.add_argument(
        "--beta_ema",
        type=float,
        default=0.0,
        help="Exponetial moving average beta.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="Gamma term of the focal loss (https://arxiv.org/pdf/1708.02002).",
    )
    parser.add_argument(
        "--lambda_loss",
        type=float,
        default=1.0,
        help="Contribution of the dense loss to the total loss.",
    )
    parser.add_argument(
        "--filter_predictions",
        type=bool_flag,
        default=False,
        help="Contribution of the dense loss to the total loss (must be in [0, 1]).",
    )
    parser.add_argument(
        "--mask_empty_classes",
        type=bool_flag,
        default=True,
        help="Wether to discard classes not present in the batch.",
    )
    parser.add_argument(
        "--bce_loss",
        type=bool_flag,
        default=False,
        help="Wether to us the BCE loss.",
    )
    parser.add_argument(
        "--batch_classes_only",
        type=bool_flag,
        default=False,
        help="Wether to only.",
    )
    return parser
    # args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    # default_params = get_default_params(args.model)
    # for name, val in default_params.items():
    #     if getattr(args, name) is None:
    #         setattr(args, name, val)

    # return args
