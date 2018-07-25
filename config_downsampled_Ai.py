import os


config = dict()

#条件名
#config["condition"] = ("/downsampled_atlas_U-Net_FL1")
config["condition"] = ("/downsampled_EUDTweight_U-Net-debug")
#config["condition"] = ("/Ai_dice_lite_unet")

#症例リストフォルダ
config["data_text_dir"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net")



#ダウンサンプル体マスクフォルダ
config["mask_dir"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net/downsampledBody/")

#パッチテキスト保存フォルダ
config["patch_dir"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net/")

#入力補外画像フォルダ
config["data_dir"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net/")

#確率アトラスフォルダ
config["atlas_dir"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net/downsampled/PA/RBF_BodyLM/")
config["atlas_dir_test"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net2/downsampled/PA/RBF_BodyLM/")

#EUDTアトラスフォルダ
config["EUDT_dir"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net/downsampled/EUDT/")


#確認用パッチ画像出力フォルダ
config["out_dir"] = os.path.abspath("D:/U-Net/temp/out" + config["condition"])

#出力モデルフォルダ
config["model_dir"] = ("D:/U-Net/out/model" + config["condition"])



#テスト補外画像フォルダ
config["data_dir_for_prediction"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net2")

#テスト補外マスク画像フォルダ
config["mask_dir_for_prediction"] = os.path.abspath("F:/f/Fukui-Ai/for_U-Net2/downsampledBody/")


#ログフォルダ
config["log_dir"] = ("D:/U-Net/out/log" + config["condition"])




#prediction使用モデル名
config["model_name"] = ("/Unet3D_1.model")
#config["model_name"] = ("/makeking_check.model")

#prediction出力フォルダ
config["test_dir"] = os.path.abspath("D:/U-Net/out"+ config["condition"])
config["test_out_dir_for_image"] = os.path.abspath(config["test_dir"] + config["model_name"])








#プレトレーニングモデル情報
config["initial_model_condition"] = ("/_for_Ai_downsampled_weighted_lite_unet")
config["initial_model_dir"] = ("D:/U-Net/out/model" + config["initial_model_condition"])
config["initial_model_name"] = ("Unet3D_5.model")



if not( os.path.isdir(config["model_dir"])):
    os.mkdir(config["model_dir"])
if not( os.path.isdir(config["out_dir"])):
    os.mkdir(config["out_dir"])
if not( os.path.isdir(config["log_dir"])):
    os.mkdir(config["log_dir"])
    os.mkdir(config["log_dir"] + "/vali_loss")
    os.mkdir(config["log_dir"] + "/JI")
if not (os.path.isdir(config["test_dir"])):
    os.mkdir(config["test_dir"])
if not (os.path.isdir(config["test_out_dir_for_image"])):
    os.mkdir(config["test_out_dir_for_image"])