import os

scene_name_list=[
    "training__Barn",
    "training__Caterpillar",
    # "training__Church",
    "training__Courthouse",
    "training__Ignatius",
    "training__Meetingroom",
    "training__Truck",
    "advanced__Auditorium",
    "advanced__Ballroom",
    "advanced__Museum",
    # "advanced__Palace",
    "advanced__Temple",
    # "advanced__Courtroom",
    "intermediate__Family",
    "intermediate__Francis",
    "intermediate__Horse",
    "intermediate__Lighthouse",
    "intermediate__M60",
    "intermediate__Panther",
    "intermediate__Playground",
    "intermediate__Train",
]
import shutil
if __name__ == "__main__":
    for scene_name in scene_name_list:
        scene_path = os.path.join("reconstructions/tempvggsfm/", scene_name)
        os.makedirs(scene_path, exist_ok=True)
        pred = os.path.join("reconstructions/tempvggsfm/",scene_name+".txt")
        pred_tar = os.path.join("reconstructions/tempvggsfm/",scene_name,"pred.txt")
        gt = os.path.join("reconstructions/tempvggsfm/",scene_name+"_gt.txt")
        gt_tar = os.path.join("reconstructions/tempvggsfm/",scene_name,"gt.txt")

        shutil.copyfile(pred,pred_tar)
        shutil.copyfile(gt,gt_tar)