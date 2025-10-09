from typing import Sequence, List, Tuple, Union, TypedDict, Type, TypeVar
from enum import Enum

MEDCLIP_BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
MEDCLIP_VIT_TYPE = 'microsoft/swin-tiny-patch4-window7-224'

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

class PromptTemplate:
    def __init__(self, pre_template: str, post_template: str):
        self.pre_template = pre_template
        self.post_template = post_template
    
    def format(self, text: str|Sequence[str]):
        if isinstance(text, str):
            text = [text]
        return [self.pre_template + t + self.post_template for t in text]

    def __call__(self, text: str|Sequence[str]):
        return self.format(text)

MedCLIP_prompts_abnormal = PromptTemplate(
    pre_template="",
    post_template=""
)

# 使用Enum统一管理类别
class MRIAbnormalCategory(Enum):
    # Coarse categories (backward-compatible)
    TUMOR = "tumor"
    HEMORRHAGE = "hemorrhage"
    EDEMA = "edema"
    ALL = "all"
    # Tumor subtypes
    TUMOR_HGG = "tumor_hgg"
    TUMOR_LGG = "tumor_lgg"
    TUMOR_MET = "tumor_met"
    TUMOR_MET_HEMORRHAGIC = "tumor_met_hemorrhagic"
    TUMOR_MENINGIOMA = "tumor_meningioma"
    TUMOR_PCNSL = "tumor_pcns_lymphoma"
    TUMOR_GBM = "tumor_gbm"
    TUMOR_BUTTERFLY = "tumor_butterfly"
    TUMOR_HIGHGRADE_MRS = "tumor_highgrade_mrs"
    # Hemorrhage subtypes
    HEM_ACUTE_ICH = "hem_acute_ich"
    HEM_CAA_MICROBLEEDS = "hem_caa_microbleeds"
    HEM_CONTUSION = "hem_contusion"
    HEM_SAH = "hem_sah"
    HEM_IVH = "hem_ivh"
    HEM_CAVERNOMA = "hem_cavernoma"
    # Edema subtypes
    EDEMA_VASOGENIC = "edema_vasogenic"
    EDEMA_CYTOTOXIC = "edema_cytotoxic"
    EDEMA_INTERSTITIAL = "edema_interstitial"
    EDEMA_PRES = "edema_pres"

# MRI脑部图像中文专业Prompt集合
# 说明：
# - 覆盖肿瘤（含转移瘤/脑膜瘤/高级别胶质瘤等）、出血（急性/亚急性/微出血/蛛网膜下腔出血/脑室出血等）与水肿（血管源性/细胞毒性/PRES/间质性）
# - 每条均包含影像序列、解剖定位与典型影像学特征，描述简洁且临床相关
# - 可结合 PromptTemplate 进行统一前后缀包装

# MRI brain prompts (English); fine-grained labels and aggregated coarse lists

# Tumor (fine-grained)
MRI_TUMOR_HGG_PROMPTS: list[str] = [
    "Axial post-contrast T1 shows an irregular ring-enhancing mass in the right frontal deep white matter with central necrosis; extensive T2/FLAIR hyperintensity indicates vasogenic edema; mild diffusion restriction along the enhancing rim; marked mass effect.",
]
MRI_TUMOR_LGG_PROMPTS: list[str] = [
    "Coronal FLAIR demonstrates diffuse subcortical hyperintensity in the left temporal lobe with mild cortical expansion; no enhancement on post-contrast T1; features suggest a low-grade glioma.",
    "Coronal FLAIR shows band-like subcortical hyperintensity in the right occipital lobe; non-enhancing on post-contrast T1; subtle cortical bulging; consistent with diffuse low-grade glioma.",
]
MRI_TUMOR_MET_PROMPTS: list[str] = [
    "Axial post-contrast T1 reveals multiple small ring-enhancing lesions at the gray–white junction of the right parietal lobe with surrounding vasogenic edema on FLAIR; favored metastases.",
]
MRI_TUMOR_MET_HEMORRHAGIC_PROMPTS: list[str] = [
    "Axial SWI demonstrates scattered blooming foci within a right parietal mass; peripheral irregular enhancement on post-contrast T1; extensive perilesional FLAIR hyperintensity; consistent with hemorrhagic metastasis.",
]
MRI_TUMOR_MENINGIOMA_PROMPTS: list[str] = [
    "Sagittal post-contrast T1 shows a dural-based extra-axial enhancing mass along the right parafalcine region with a dural tail; T2 iso- to hypointense; mild mass effect on the adjacent cortex; typical for meningioma.",
]
MRI_TUMOR_PCNSL_PROMPTS: list[str] = [
    "Axial DWI shows marked diffusion restriction in a nodular lesion in the left basal ganglia; homogeneous enhancement on post-contrast T1; T2 hypointensity; consistent with primary CNS lymphoma.",
]
MRI_TUMOR_GBM_PROMPTS: list[str] = [
    "Coronal post-contrast T1 shows a heterogeneous enhancing mass centered in the left insula; extensive peritumoral FLAIR hyperintensity; elevated rCBV on DSC perfusion; favors glioblastoma.",
]
MRI_TUMOR_BUTTERFLY_PROMPTS: list[str] = [
    "Axial T2 demonstrates a butterfly-shaped infiltrative lesion crossing the splenium of the corpus callosum; irregular peripheral enhancement on post-contrast T1; extensive surrounding edema; suggest high-grade infiltrative glioma.",
]
MRI_TUMOR_HIGHGRADE_MRS_PROMPTS: list[str] = [
    "Single-voxel MRS over the left frontal mass shows markedly increased Cho/NAA ratio; together with irregular enhancement and perilesional edema, indicates a high-grade neoplasm.",
]

# Hemorrhage (fine-grained)
MRI_HEM_ACUTE_ICH_PROMPTS: list[str] = [
    "Axial SWI shows pronounced susceptibility blooming in an acute hematoma centered in the left basal ganglia; surrounding FLAIR hyperintensity and mild midline shift indicate edema and mass effect.",
]
MRI_HEM_CAA_MICROBLEEDS_PROMPTS: list[str] = [
    "Axial T2*/SWI reveals numerous punctate cortical–subcortical microbleeds in bilateral occipital lobes and cerebellum; mild surrounding FLAIR hyperintensity; pattern consistent with cerebral amyloid angiopathy.",
]
MRI_HEM_CONTUSION_PROMPTS: list[str] = [
    "Coronal FLAIR shows right frontotemporal contusion with patchy T1 hyperintense subacute blood products; surrounding vasogenic edema and mass effect are present.",
]
MRI_HEM_SAH_PROMPTS: list[str] = [
    "Axial FLAIR and T1 demonstrate laminar high signal in the bilateral Sylvian fissures and cortical sulci; SWI shows corresponding low-signal deposition; findings suggest subarachnoid hemorrhage with Sylvian predominance.",
]
MRI_HEM_IVH_PROMPTS: list[str] = [
    "Axial T1 and SWI show layered blood products within the occipital horns of the lateral ventricles; periventricular FLAIR hyperintensity indicates transependymal CSF flow; consistent with intraventricular hemorrhage.",
]
MRI_HEM_CAVERNOMA_PROMPTS: list[str] = [
    "Sagittal T2* demonstrates a focal blooming lesion in the right thalamus with a hemosiderin rim and minimal surrounding FLAIR hyperintensity; compatible with cavernous malformation with prior hemorrhage.",
]

# Edema (fine-grained)
MRI_EDEMA_VASOGENIC_PROMPTS: list[str] = [
    "Axial FLAIR shows extensive vasogenic edema in the left parietal lobe involving subcortical white matter and tracking along fiber tracts; no diffusion restriction; likely peritumoral edema.",
]
MRI_EDEMA_CYTOTOXIC_PROMPTS: list[str] = [
    "Axial DWI/ADC demonstrates marked diffusion restriction in the head of the right caudate; minimal early FLAIR change; consistent with acute ischemia with cytotoxic edema.",
]
MRI_EDEMA_INTERSTITIAL_PROMPTS: list[str] = [
    "Coronal FLAIR shows periventricular linear hyperintensity along the ependyma with mild third ventricular enlargement; features of transependymal CSF flow due to obstructive hydrocephalus.",
]
MRI_EDEMA_PRES_PROMPTS: list[str] = [
    "Axial FLAIR and T2 demonstrate symmetric subcortical hyperintensity in the parieto-occipital lobes; negative for diffusion restriction on DWI; no enhancement; in the clinical context of blood pressure fluctuation, favors PRES-related vasogenic edema.",
]

# Coarse category lists (aggregated for backward compatibility)
MRI_TUMOR_PROMPTS: list[str] = (
    MRI_TUMOR_HGG_PROMPTS
    + MRI_TUMOR_LGG_PROMPTS
    + MRI_TUMOR_MET_PROMPTS
    + MRI_TUMOR_MET_HEMORRHAGIC_PROMPTS
    + MRI_TUMOR_MENINGIOMA_PROMPTS
    + MRI_TUMOR_PCNSL_PROMPTS
    + MRI_TUMOR_GBM_PROMPTS
    + MRI_TUMOR_BUTTERFLY_PROMPTS
    + MRI_TUMOR_HIGHGRADE_MRS_PROMPTS
)

MRI_HEMORRHAGE_PROMPTS: list[str] = (
    MRI_HEM_ACUTE_ICH_PROMPTS
    + MRI_HEM_CAA_MICROBLEEDS_PROMPTS
    + MRI_HEM_CONTUSION_PROMPTS
    + MRI_HEM_SAH_PROMPTS
    + MRI_HEM_IVH_PROMPTS
    + MRI_HEM_CAVERNOMA_PROMPTS
)

MRI_EDEMA_PROMPTS: list[str] = (
    MRI_EDEMA_VASOGENIC_PROMPTS
    + MRI_EDEMA_CYTOTOXIC_PROMPTS
    + MRI_EDEMA_INTERSTITIAL_PROMPTS
    + MRI_EDEMA_PRES_PROMPTS
)

# 聚合列表（便于一次性取用全部异常描述）
MRI_ABNORMAL_PROMPTS: list[str] = (
    MRI_TUMOR_PROMPTS + MRI_HEMORRHAGE_PROMPTS + MRI_EDEMA_PROMPTS
)

# 便捷映射
ALL_MRI_PROMPTS: dict[Union[MRIAbnormalCategory, str], list[str]] = {
    # Coarse
    MRIAbnormalCategory.TUMOR: MRI_TUMOR_PROMPTS,
    MRIAbnormalCategory.HEMORRHAGE: MRI_HEMORRHAGE_PROMPTS,
    MRIAbnormalCategory.EDEMA: MRI_EDEMA_PROMPTS,
    MRIAbnormalCategory.ALL: MRI_ABNORMAL_PROMPTS,
    # Tumor subtypes
    MRIAbnormalCategory.TUMOR_HGG: MRI_TUMOR_HGG_PROMPTS,
    MRIAbnormalCategory.TUMOR_LGG: MRI_TUMOR_LGG_PROMPTS,
    MRIAbnormalCategory.TUMOR_MET: MRI_TUMOR_MET_PROMPTS,
    MRIAbnormalCategory.TUMOR_MET_HEMORRHAGIC: MRI_TUMOR_MET_HEMORRHAGIC_PROMPTS,
    MRIAbnormalCategory.TUMOR_MENINGIOMA: MRI_TUMOR_MENINGIOMA_PROMPTS,
    MRIAbnormalCategory.TUMOR_PCNSL: MRI_TUMOR_PCNSL_PROMPTS,
    MRIAbnormalCategory.TUMOR_GBM: MRI_TUMOR_GBM_PROMPTS,
    MRIAbnormalCategory.TUMOR_BUTTERFLY: MRI_TUMOR_BUTTERFLY_PROMPTS,
    MRIAbnormalCategory.TUMOR_HIGHGRADE_MRS: MRI_TUMOR_HIGHGRADE_MRS_PROMPTS,
    # Hemorrhage subtypes
    MRIAbnormalCategory.HEM_ACUTE_ICH: MRI_HEM_ACUTE_ICH_PROMPTS,
    MRIAbnormalCategory.HEM_CAA_MICROBLEEDS: MRI_HEM_CAA_MICROBLEEDS_PROMPTS,
    MRIAbnormalCategory.HEM_CONTUSION: MRI_HEM_CONTUSION_PROMPTS,
    MRIAbnormalCategory.HEM_SAH: MRI_HEM_SAH_PROMPTS,
    MRIAbnormalCategory.HEM_IVH: MRI_HEM_IVH_PROMPTS,
    MRIAbnormalCategory.HEM_CAVERNOMA: MRI_HEM_CAVERNOMA_PROMPTS,
    # Edema subtypes
    MRIAbnormalCategory.EDEMA_VASOGENIC: MRI_EDEMA_VASOGENIC_PROMPTS,
    MRIAbnormalCategory.EDEMA_CYTOTOXIC: MRI_EDEMA_CYTOTOXIC_PROMPTS,
    MRIAbnormalCategory.EDEMA_INTERSTITIAL: MRI_EDEMA_INTERSTITIAL_PROMPTS,
    MRIAbnormalCategory.EDEMA_PRES: MRI_EDEMA_PRES_PROMPTS,
}
# 兼容历史字符串key（尽量不破坏外部依赖）
ALL_MRI_PROMPTS.update({
    # Coarse
    "tumor": MRI_TUMOR_PROMPTS,
    "hemorrhage": MRI_HEMORRHAGE_PROMPTS,
    "edema": MRI_EDEMA_PROMPTS,
    "all": MRI_ABNORMAL_PROMPTS,
    # Tumor subtypes
    "tumor_hgg": MRI_TUMOR_HGG_PROMPTS,
    "tumor_lgg": MRI_TUMOR_LGG_PROMPTS,
    "tumor_met": MRI_TUMOR_MET_PROMPTS,
    "tumor_met_hemorrhagic": MRI_TUMOR_MET_HEMORRHAGIC_PROMPTS,
    "tumor_meningioma": MRI_TUMOR_MENINGIOMA_PROMPTS,
    "tumor_pcns_lymphoma": MRI_TUMOR_PCNSL_PROMPTS,
    "tumor_gbm": MRI_TUMOR_GBM_PROMPTS,
    "tumor_butterfly": MRI_TUMOR_BUTTERFLY_PROMPTS,
    "tumor_highgrade_mrs": MRI_TUMOR_HIGHGRADE_MRS_PROMPTS,
    # Hemorrhage subtypes
    "hem_acute_ich": MRI_HEM_ACUTE_ICH_PROMPTS,
    "hem_caa_microbleeds": MRI_HEM_CAA_MICROBLEEDS_PROMPTS,
    "hem_contusion": MRI_HEM_CONTUSION_PROMPTS,
    "hem_sah": MRI_HEM_SAH_PROMPTS,
    "hem_ivh": MRI_HEM_IVH_PROMPTS,
    "hem_cavernoma": MRI_HEM_CAVERNOMA_PROMPTS,
    # Edema subtypes
    "edema_vasogenic": MRI_EDEMA_VASOGENIC_PROMPTS,
    "edema_cytotoxic": MRI_EDEMA_CYTOTOXIC_PROMPTS,
    "edema_interstitial": MRI_EDEMA_INTERSTITIAL_PROMPTS,
    "edema_pres": MRI_EDEMA_PRES_PROMPTS,
})


def get_mri_prompts(category: Union[MRIAbnormalCategory, str] = MRIAbnormalCategory.ALL, template: PromptTemplate | None = None) -> list[str]:
    """
    按类别获取MRI中文prompt列表，并可选用PromptTemplate进行前后缀包装。
    category: MRIAbnormalCategory | "tumor" | "hemorrhage" | "edema" | "all"
    template: 若提供，将对每条文本进行 pre+text+post 的统一格式化
    """
    if isinstance(category, str):
        try:
            category_enum = MRIAbnormalCategory(category.lower())
        except ValueError:
            category_enum = MRIAbnormalCategory.ALL
    else:
        category_enum = category

    texts = ALL_MRI_PROMPTS.get(category_enum, MRI_ABNORMAL_PROMPTS)
    if template is None:
        return texts
    return template.format(texts)
