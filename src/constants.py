from pathlib import Path

DATASET_ROOT_DIR = 'data'
CACHE_DIR = 'cache' # last run cache info
PRETRAINED_MODEL_DIR = 'cache/models/pretrained'
CONFIG_DIR = 'configs'
DATASET_CONFIG_DIR = 'configs/dataset'
PIPELINE_CONFIG_DIR = 'configs/pipeline'
SINGLE_CONFIG_DIR = 'configs/single'
RUN_DIR = 'runs'
OUTPUT_DIR = 'outputs'
TEMP_DIR = 'TEMP'

class ProjectFilenameEnv:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance
    
    @property
    def output_dir(self) -> Path:
        return Path(OUTPUT_DIR)
    @property
    def temp_dir(self) -> Path:
        return Path(TEMP_DIR)
    @property
    def run_dir(self) -> Path:
        return Path(RUN_DIR)
    @property
    def config_dir(self) -> Path:
        return Path(CONFIG_DIR)
    @property
    def dataset_config_dir(self) -> Path:
        return Path(DATASET_CONFIG_DIR)
    @property
    def pipeline_config_dir(self) -> Path:
        return Path(PIPELINE_CONFIG_DIR)
    @property
    def single_config_dir(self) -> Path:
        return Path(SINGLE_CONFIG_DIR)
    @property
    def dataset_root_dir(self) -> Path:
        return Path(DATASET_ROOT_DIR)
    @property
    def cache_dir(self) -> Path:
        return Path(CACHE_DIR)
    @property
    def pretrained_model_dir(self) -> Path:
        return Path(PRETRAINED_MODEL_DIR)

class TrainOutputFilenameEnv:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    # image
    OUTPUT_TRAIN_LOSS_FILENAME = '{train_dir}/train_epoch_loss.png'
    OUTPUT_VALID_LOSS_FILENAME = '{train_dir}/valid_epoch_loss.png'
    # checkpoint
    OUTPUT_LAST_MODEL_FILENAME = '{train_dir}/weights/last.pt'
    OUTPUT_BEST_MODEL_FILENAME = '{train_dir}/weights/best.pt'
    OUTPUT_LAST_EXT_MODEL_FILENAME = '{train_dir}/weights/last.ext.pt'
    OUTPUT_BEST_EXT_MODEL_FILENAME = '{train_dir}/weights/best.ext.pt'
    OUTPUT_TEMP_MODEL_FILENAME = '{train_dir}/weights/{model}-{epoch}of{num_epochs}.pt'
    OUTPUT_TEMP_EXT_MODEL_FILENAME = '{train_dir}/weights/{model}-{epoch}of{num_epochs}.ext.pt'
    # csv
    OUTPUT_TRAIN_LOSS_DETAILS_FILENAME = '{train_dir}/train_epoch_loss.csv'
    OUTPUT_VALID_LOSS_DETAILS_FILENAME = '{train_dir}/valid_epoch_loss.csv'
    OUTPUT_LOSS_DETAIL_FILENAME = '{train_dir}/loss.csv' # combine train and valid
    # recovery
    RECOVERY_DIR = '{train_dir}/recovery'

    def register(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        return self

    @property
    def output_train_loss_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_LOSS_FILENAME.format(train_dir=self.train_dir))
    @property
    def output_valid_loss_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_LOSS_FILENAME.format(train_dir=self.train_dir))   
    @property
    def output_loss_detail_filename(self) -> Path:
        return Path(self.OUTPUT_LOSS_DETAIL_FILENAME.format(train_dir=self.train_dir))
    @property
    def output_last_model_filename(self) -> Path:
        return Path(self.OUTPUT_LAST_MODEL_FILENAME.format(train_dir=self.train_dir))
    @property
    def output_best_model_filename(self) -> Path:
        return Path(self.OUTPUT_BEST_MODEL_FILENAME.format(train_dir=self.train_dir))
    @property   
    def output_last_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_LAST_EXT_MODEL_FILENAME.format(train_dir=self.train_dir))
    @property
    def output_best_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_BEST_EXT_MODEL_FILENAME.format(train_dir=self.train_dir))
    @property
    def output_temp_model_filename(self) -> Path:
        return Path(self.OUTPUT_TEMP_MODEL_FILENAME.format(train_dir=self.train_dir, model=self.model, epoch=self.epoch, num_epochs=self.num_epochs))
    @property
    def output_temp_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_TEMP_EXT_MODEL_FILENAME.format(train_dir=self.train_dir, model=self.model, epoch=self.epoch, num_epochs=self.num_epochs))
    @property
    def output_train_loss_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_LOSS_DETAILS_FILENAME.format(train_dir=self.train_dir))
    @property   
    def output_valid_loss_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_LOSS_DETAILS_FILENAME.format(train_dir=self.train_dir))
    @property
    def recovery_dir(self) -> Path:
        return Path(self.RECOVERY_DIR.format(train_dir=self.train_dir))

class InferenceOutputFilenameEnv:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    OUTPUT_RESULT_FILENAME = '{infer_dir}/{input_filename}'

    def register(self, infer_dir: Path, **kwargs):
        setattr(self, 'infer_dir', infer_dir)
        for k,v in kwargs.items():
            setattr(self, k, v)

    @property
    def output_result_filename(self, input_filename: str) -> Path:
        return Path(self.OUTPUT_RESULT_FILENAME.format(infer_dir=self.infer_dir, input_filename=input_filename))
