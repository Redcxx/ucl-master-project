from ml.options.base import BaseTrainOptions, BaseInferenceOptions


class AlacGANInferenceOptions(BaseInferenceOptions):

    def __init__(self):
        super().__init__()
        self.input_images_path = 'inference_images'
        self.output_images_path = 'output_images'
        self.image_size = 512
        self.a_to_b = True
        self.batch_size = 8

    @property
    def tag(self):
        return 'alac_gan_inference'

    @property
    def inference_run_id(self):
        return 'alacGAN-train-2022-06-29-Wednesday-17h-23m-26s'


class AlacGANTrainOptions(BaseTrainOptions):

    @property
    def tag(self):
        return 'alacGAN-train'

    def __init__(self):
        super().__init__()

        # self.run_id = r'alacGAN-train-2022-06-28-Tuesday-19h-02m-04s'

        # Training
        self.batch_size = 8
        self.start_epoch = 1
        self.end_epoch = 30
        self.eval_freq = 1
        self.log_freq = 1
        self.save_freq = 1
        self.batch_log_freq = 100

        self.image_size = 512

        # Dataset
        self.dataset_root = './alacgan_colorization_data'
        self.a_to_b = True

        # Optimizer
        self.lr = 0.0001

        # Backbones checkpoint
        self.VGG16_PATH = 'vgg16-397923af.pth'
        self.I2V_PATH = 'i2v.pth'
