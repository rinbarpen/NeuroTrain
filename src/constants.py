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
    OUTPUT_EPOCH_LOSS_IMAGE_FILENAME = '{train_dir}/epoch_loss.png' # combine train and valid
    OUTPUT_TRAIN_EPOCH_LOSS_IMAGE_FILENAME = '{train_dir}/train/train_epoch_loss.png'
    OUTPUT_VALID_EPOCH_LOSS_IMAGE_FILENAME = '{train_dir}/valid/valid_epoch_loss.png'
    OUTPUT_TRAIN_STEP_LOSS_IMAGE_FILENAME = '{train_dir}/train/train_step_loss.png'
    # checkpoint
    OUTPUT_LAST_MODEL_FILENAME = '{train_dir}/weights/last.pt'
    OUTPUT_BEST_MODEL_FILENAME = '{train_dir}/weights/best.pt'
    OUTPUT_LAST_EXT_MODEL_FILENAME = '{train_dir}/weights/last.ext.pt'
    OUTPUT_BEST_EXT_MODEL_FILENAME = '{train_dir}/weights/best.ext.pt'
    OUTPUT_TEMP_MODEL_FILENAME = '{train_dir}/weights/{model}-{epoch}of{num_epochs}.pt'
    OUTPUT_TEMP_EXT_MODEL_FILENAME = '{train_dir}/weights/{model}-{epoch}of{num_epochs}.ext.pt'
    OUTPUT_STOP_MODEL_FILENAME = '{train_dir}/weights/stop.pt'
    OUTPUT_STOP_EXT_MODEL_FILENAME = '{train_dir}/weights/stop.ext.pt'
    # csv
    OUTPUT_TRAIN_EPOCH_LOSS_DETAILS_FILENAME = '{train_dir}/train/train_epoch_loss.csv'
    OUTPUT_VALID_EPOCH_LOSS_DETAILS_FILENAME = '{train_dir}/valid/valid_epoch_loss.csv'
    # metric csv
    OUTPUT_TRAIN_ALL_METRIC_DETAILS_FILENAME = '{train_dir}/train/train_all_metric.csv'
    OUTPUT_VALID_ALL_METRIC_DETAILS_FILENAME = '{train_dir}/valid/valid_all_metric.csv'
    OUTPUT_TRAIN_MEAN_METRIC_DETAILS_FILENAME = '{train_dir}/train/train_mean_metric.csv'
    OUTPUT_VALID_MEAN_METRIC_DETAILS_FILENAME = '{train_dir}/valid/valid_mean_metric.csv'
    OUTPUT_TRAIN_STD_METRIC_DETAILS_FILENAME = '{train_dir}/train/train_std_metric.csv'
    OUTPUT_VALID_STD_METRIC_DETAILS_FILENAME = '{train_dir}/valid/valid_std_metric.csv'
    OUTPUT_TRAIN_MEAN_METRIC_BY_CLASS_DETAILS_FILENAME = '{train_dir}/train/{class_label}/train_mean_metric.csv'
    OUTPUT_VALID_MEAN_METRIC_BY_CLASS_DETAILS_FILENAME = '{train_dir}/valid/{class_label}/valid_mean_metric.csv'
    OUTPUT_TRAIN_STD_METRIC_BY_CLASS_DETAILS_FILENAME = '{train_dir}/train/{class_label}/train_std_metric.csv'
    OUTPUT_VALID_STD_METRIC_BY_CLASS_DETAILS_FILENAME = '{train_dir}/valid/{class_label}/valid_std_metric.csv'
    OUTPUT_TRAIN_ALL_METRIC_BY_CLASS_DETAILS_FILENAME = '{train_dir}/train/{class_label}/train_all_metric.csv'
    OUTPUT_VALID_ALL_METRIC_BY_CLASS_DETAILS_FILENAME = '{train_dir}/valid/{class_label}/valid_all_metric.csv'
    # metric image
    OUTPUT_TRAIN_ALL_METRIC_IMAGE_FILENAME = '{train_dir}/train/train_all_metric.png'
    OUTPUT_VALID_ALL_METRIC_IMAGE_FILENAME = '{train_dir}/valid/valid_all_metric.png'
    OUTPUT_TRAIN_MEAN_STD_METRIC_IMAGE_FILENAME = '{train_dir}/train/train_mean_std_metric.png'
    OUTPUT_VALID_MEAN_STD_METRIC_IMAGE_FILENAME = '{train_dir}/valid/valid_mean_std_metric.png'
    OUTPUT_TRAIN_MEAN_STD_METRIC_BY_CLASS_IMAGE_FILENAME = '{train_dir}/train/train_mean_std_metric_by_class.png'
    OUTPUT_VALID_MEAN_STD_METRIC_BY_CLASS_IMAGE_FILENAME = '{train_dir}/valid/valid_mean_std_metric_by_class.png'
    OUTPUT_TRAIN_ALL_METRIC_BY_CLASS_IMAGE_FILENAME = '{train_dir}/train/train_all_metric_by_class.png'
    OUTPUT_VALID_ALL_METRIC_BY_CLASS_IMAGE_FILENAME = '{train_dir}/valid/valid_all_metric_by_class.png'
    # step csv
    OUTPUT_TRAIN_STEP_LOSS_FILENAME = '{train_dir}/train/train_step_loss.csv'
    OUTPUT_TRAIN_STEP_LOSS_IMAGE_FILENAME = '{train_dir}/train/train_step_loss.png'
    # lr csv
    OUTPUT_LR_BY_EPOCH_FILENAME = '{train_dir}/lr_by_epoch.csv'
    OUTPUT_LR_BY_STEP_FILENAME = '{train_dir}/lr_by_step.csv'
    # lr image
    OUTPUT_LR_BY_EPOCH_IMAGE_FILENAME = '{train_dir}/lr_by_epoch.png'
    OUTPUT_LR_BY_STEP_IMAGE_FILENAME = '{train_dir}/lr_by_step.png'
    # recovery
    RECOVERY_DIR = '{train_dir}/recovery'
    STOP_FLAG_FILENAME = '{train_dir}/STOP'

    def register(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        return self

    def prepare_dir(self):
        root_dir = Path(getattr(self, "train_dir")) 
        (root_dir / "train").mkdir(exist_ok=True, parents=True)
        (root_dir / "valid").mkdir(exist_ok=True, parents=True)
        (root_dir / "weights").mkdir(exist_ok=True, parents=True)
        (root_dir / "recovery").mkdir(exist_ok=True, parents=True)

        # 为每个类别创建目录（如果 class_labels 存在的话）
        class_labels = getattr(self, "class_labels", None)
        if class_labels:
            for class_label in class_labels:
                (root_dir / "train" / class_label).mkdir(exist_ok=True, parents=True)
                (root_dir / "valid" / class_label).mkdir(exist_ok=True, parents=True)

    # loss image files
    @property
    def output_epoch_loss_image_filename(self) -> Path:
        return Path(self.OUTPUT_EPOCH_LOSS_IMAGE_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_epoch_loss_image_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_EPOCH_LOSS_IMAGE_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_epoch_loss_image_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_EPOCH_LOSS_IMAGE_FILENAME.format(train_dir=getattr(self, "train_dir")))   
    @property
    # loss csv files
    def output_train_step_loss_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_STEP_LOSS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_step_loss_image_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_STEP_LOSS_IMAGE_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_epoch_loss_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_EPOCH_LOSS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_epoch_loss_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_EPOCH_LOSS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    # lr csv files
    @property
    def output_lr_by_epoch_filename(self) -> Path:
        return Path(self.OUTPUT_LR_BY_EPOCH_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_lr_by_step_filename(self) -> Path:
        return Path(self.OUTPUT_LR_BY_STEP_FILENAME.format(train_dir=getattr(self, "train_dir")))
    # lr image files
    @property
    def output_lr_by_epoch_image_filename(self) -> Path:
        return Path(self.OUTPUT_LR_BY_EPOCH_IMAGE_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_lr_by_step_image_filename(self) -> Path:
        return Path(self.OUTPUT_LR_BY_STEP_IMAGE_FILENAME.format(train_dir=getattr(self, "train_dir")))
    # metric csv files
    @property
    def output_train_all_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_ALL_METRIC_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_all_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_ALL_METRIC_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_mean_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_MEAN_METRIC_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_mean_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_MEAN_METRIC_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_std_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_STD_METRIC_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_std_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_STD_METRIC_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_mean_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_MEAN_METRIC_BY_CLASS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_mean_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_MEAN_METRIC_BY_CLASS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_std_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_STD_METRIC_BY_CLASS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_std_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_STD_METRIC_BY_CLASS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_train_all_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_TRAIN_ALL_METRIC_BY_CLASS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_valid_all_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_VALID_ALL_METRIC_BY_CLASS_DETAILS_FILENAME.format(train_dir=getattr(self, "train_dir")))
    # model
    @property
    def output_last_model_filename(self) -> Path:
        return Path(self.OUTPUT_LAST_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_best_model_filename(self) -> Path:
        return Path(self.OUTPUT_BEST_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property   
    def output_last_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_LAST_EXT_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_best_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_BEST_EXT_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_temp_model_filename(self) -> Path:
        return Path(self.OUTPUT_TEMP_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir"), model=getattr(self, "model"), epoch=getattr(self, "epoch"), num_epochs=getattr(self, "num_epochs")))
    @property
    def output_temp_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_TEMP_EXT_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir"), model=getattr(self, "model"), epoch=getattr(self, "epoch"), num_epochs=getattr(self, "num_epochs")))
    # recovery
    @property
    def recovery_dir(self) -> Path:
        return Path(self.RECOVERY_DIR.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_stop_model_filename(self) -> Path:
        return Path(self.OUTPUT_STOP_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def output_stop_ext_model_filename(self) -> Path:
        return Path(self.OUTPUT_STOP_EXT_MODEL_FILENAME.format(train_dir=getattr(self, "train_dir")))
    @property
    def stop_flag_file(self) -> Path:
        return Path(self.STOP_FLAG_FILENAME.format(train_dir=getattr(self, "train_dir")))

class TestOutputFilenameEnv:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance
    
    # csv files
    OUTPUT_TEST_ALL_METRIC_DETAILS_FILENAME = '{test_dir}/test_all_metric.csv'
    OUTPUT_TEST_MEAN_METRIC_DETAILS_FILENAME = '{test_dir}/test_mean_metric.csv'
    OUTPUT_TEST_STD_METRIC_DETAILS_FILENAME = '{test_dir}/test_std_metric.csv'
    OUTPUT_TEST_MEAN_METRIC_BY_CLASS_DETAILS_FILENAME = '{test_dir}/{class_label}/test_mean_metric_by_class.csv'
    OUTPUT_TEST_STD_METRIC_BY_CLASS_DETAILS_FILENAME = '{test_dir}/{class_label}/test_std_metric_by_class.csv'
    OUTPUT_TEST_ALL_METRIC_BY_CLASS_DETAILS_FILENAME = '{test_dir}/{class_label}/test_all_metric_by_class.csv'

    # metric image files
    OUTPUT_TEST_METRIC_IMAGE_FILENAME = '{test_dir}/metric.png'
    OUTPUT_TEST_METRIC_BY_CLASS_IMAGE_FILENAME = '{test_dir}/{class_label}/metric_by_class.png'

    def register(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def prepare_dir(self):
        test_dir = getattr(self, "test_dir")
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # 为每个类别创建目录（如果 class_labels 存在的话）
        class_labels = getattr(self, "class_labels", None)
        if class_labels:
            for class_label in class_labels:
                (test_dir / class_label).mkdir(exist_ok=True)

    @property
    def output_test_all_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_ALL_METRIC_DETAILS_FILENAME.format(test_dir=getattr(self, "test_dir")))
    @property
    def output_test_mean_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_MEAN_METRIC_DETAILS_FILENAME.format(test_dir=getattr(self, "test_dir")))
    @property
    def output_test_std_metric_details_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_STD_METRIC_DETAILS_FILENAME.format(test_dir=getattr(self, "test_dir")))
    @property
    def output_test_mean_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_MEAN_METRIC_BY_CLASS_DETAILS_FILENAME.format(test_dir=getattr(self, "test_dir"), class_label=getattr(self, "class_label")))
    @property
    def output_test_std_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_STD_METRIC_BY_CLASS_DETAILS_FILENAME.format(test_dir=getattr(self, "test_dir"), class_label=getattr(self, "class_label")))
    @property
    def output_test_all_metric_by_class_details_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_ALL_METRIC_BY_CLASS_DETAILS_FILENAME.format(test_dir=getattr(self, "test_dir"), class_label=getattr(self, "class_label")))
    
    # metric image files
    @property
    def output_test_metric_image_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_METRIC_IMAGE_FILENAME.format(test_dir=getattr(self, "test_dir")))
    @property
    def output_test_metric_by_class_image_filename(self) -> Path:
        return Path(self.OUTPUT_TEST_METRIC_BY_CLASS_IMAGE_FILENAME.format(test_dir=getattr(self, "test_dir"), class_label=getattr(self, "class_label")))


class InferenceOutputFilenameEnv:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance

    OUTPUT_RESULT_FILENAME = '{infer_dir}/{input_filename}'

    def register(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    @property
    def output_result_filename(self) -> Path:
        return Path(
            self.OUTPUT_RESULT_FILENAME.format(
                infer_dir=getattr(self, 'infer_dir'), 
                input_filename=getattr(self, 'input_filename')))
