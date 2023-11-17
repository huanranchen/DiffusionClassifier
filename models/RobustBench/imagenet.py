from robustbench.utils import load_model
from robustbench.model_zoo.imagenet import normalize_model


def Wong2020Fast(pretrained=True):
    return load_model(model_name="Wong2020Fast", dataset="imagenet", threat_model="Linf")


def Engstrom2019Robustness(pretrained=True):
    return load_model(model_name="Engstrom2019Robustness", dataset="imagenet", threat_model="Linf")


def Salman2020Do_R50(pretrained=True):
    return load_model(model_name="Salman2020Do_R50", dataset="imagenet", threat_model="Linf")


def Salman2020Do_R18(pretrained=True):
    return load_model(model_name="Salman2020Do_R18", dataset="imagenet", threat_model="Linf")


def Salman2020Do_50_2(pretrained=True):
    return load_model(model_name="Salman2020Do_50_2", dataset="imagenet", threat_model="Linf")


def Debenedetti2022Light_XCiT_S12(pretrained=True):
    return load_model(model_name="Debenedetti2022Light_XCiT-S12", dataset="imagenet", threat_model="Linf")


def Debenedetti2022Light_XCiT_M12(pretrained=True):
    return load_model(model_name="Debenedetti2022Light_XCiT-M12", dataset="imagenet", threat_model="Linf")


def Debenedetti2022Light_XCiT_L12(pretrained=True):
    return load_model(model_name="Debenedetti2022Light_XCiT-L12", dataset="imagenet", threat_model="Linf")