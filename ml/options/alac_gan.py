from ml.options.base import BaseTrainOptions, BaseInferenceOptions


class AlacGANInferenceOptions(BaseInferenceOptions):

    def __init__(self):
        super().__init__()
        # self.input_images_path = r'./colorization/test'
        self.input_images_path = r'colorization/test'
        self.output_images_path = 'noghost_colorization_output'
        self.image_size = 512
        self.a_to_b = True
        self.batch_size = 8
        self.num_workers = 4
        self.hint_mask = True
        self.hint_multiplier = 1
        self.custom_color = None  # red: (235, 64, 52), yellow: (252, 186, 3), blue: (66, 135, 245)
        self.limit = None  # limit on the inference dataset

    @property
    def tag(self):
        return 'alac_gan_noghost_inference'

    @property
    def inference_run_id(self):
        return 'alacGAN-train-2022-07-02-Saturday-12h-25m-53s'  # finetuned
        # return 'alacGAN-train-2022-06-29-Wednesday-17h-23m-26s'  # pretrained

    @property
    def inference_run_tag(self):
        return 'final'


class AlacGANTrainOptions(BaseTrainOptions):

    @property
    def tag(self):
        return 'alacGAN-train-sketch'

    def __init__(self):
        super().__init__()

        # Model
        self.use_hint = True

        # Training
        self.batch_size = 8
        self.start_epoch = 1
        self.end_epoch = 250
        self.eval_freq = 25
        self.log_freq = 1
        self.save_freq = 50
        self.batch_log_freq = 0
        # self.resume_ckpt_file = 'alacGAN-train-2022-06-29-Wednesday-17h-23m-26s_latest.ckpt'

        # Dataset
        self.image_size = 512
        self.dataset_root = './sketch_simplification'
        self.a_to_b = False
        self.make_fake_hint = True
        self.in_channels = 1
        self.out_channels = 3

        # Optimizer
        self.lr = 0.00001

        # Scheduler
        self.scheduler_step_size = 200
        self.scheduler_gamma = 0.1

        # Backbones checkpoint
        self.VGG16_PATH = 'vgg16-397923af.pth'
        self.I2V_PATH = 'i2v.pth'
