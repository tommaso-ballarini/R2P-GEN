categories=[
    'bag',
    'book',
    'bottle',
    'bowl',
    'clothe',
    'cup',
    'decoration',
    'headphone',
    'pillow',
    'plant',
    'plate',
    'remote',
    'retail',
    'telephone',
    'tie',
    'towel',
    'toy',
    'tro_bag',
    'tumbler',
    'umbrella',
    'veg'
]

myvlm_category_dict = {
    "object": ['asian_doll', 'boy_funko_pop', 'bull', 'cat_statue', 'ceramic_head', 'chicken_bean_bag', 'colorful_teapot',
                'dangling_child', 'elephant_sphere', 'elephant_statue', 'espresso_cup', 'gengar_toy', 'gold_pineapple',
                'green_doll', 'iverson_funko_pop', 'maeve_dog', 'minion_toy', 'rabbit_toy', 'red_chicken', 'red_piggy_bank',
                'robot_toy', 'running_shoes', 'sheep_pillow', 'sheep_plush', 'sheep_toy', 'skulls_mug', 'small_penguin'],
    "pet animal": ['billy_dog', 'my_cat']
    }

yollava_category_dict = {
            "person": ['ciin', 'denisdang', 'khanhvy', 'oong', 'phuc-map', 'thao', 'thuytien', 'viruss', 'yuheng', "willinvietnam"],
            "building": ['chua-thien-mu', 'nha-tho-hanoi', 'nha-tho-hcm', 'thap-but', 'thap-cham'],
            "cartoon character": ['dug', 'fire', 'marie-cat', 'toodles-galore', 'water'],
            "pet animal": ['bo', 'butin', 'henry', 'mam', 'mydieu'],
            "object": ["shiba-yellow", 'pusheen-cup', "neurips-cup", "tokyo-keyboard", "cat-cup", "brown-duck", "lamb",
                       "duck-banana", "shiba-black", "pig-cup", "shiba-sleep", "yellow-duck", "elephant", "shiba-gray", "dragon"]
        }

osc_reverse_category_dict = {
    'bag':'bag',
    'book':'book',
    'bottle':'bottle',
    'bowl':'bowl',
    'clothe':'clothe',
    'cup':'cup',
    'decoration':'decoration',
    'headphone':'headphone',
    'pillow':'pillow',
    'plant':'plant',
    'plate':'plate',
    'remote':'remote',
    'retail':'retail product',
    'telephone':'cell phone',
    'tie':'tie',
    'towel':'towel',
    'toy':'toy',
    'tro_bag':'trolley bag',
    'tumbler':"glass",
    'umbrella':'umbrella',
    'veg':'vegetable'
}

yollava_reverse_category_dict = {
    'ciin': 'person',
    'denisdang': 'person',
    'khanhvy': 'person',
    'oong': 'person',
    'phuc-map': 'person',
    'thao': 'person',
    'thuytien': 'person',
    'viruss': 'person',
    'yuheng': 'person',
    'willinvietnam': 'person',
    'chua-thien-mu': 'building',
    'nha-tho-hanoi': 'building',
    'nha-tho-hcm': 'building',
    'thap-but': 'building',
    'thap-cham': 'building',
    'dug': 'cartoon character',
    'fire': 'cartoon character',
    'marie-cat': 'cartoon character',
    'toodles-galore': 'cartoon character',
    'water': 'cartoon character',
    'bo': 'pet animal',
    'butin': 'pet animal',
    'henry': 'pet animal',
    'mam': 'pet animal',
    'mydieu': 'pet animal',
    'shiba-yellow': 'toy',
    'pusheen-cup': 'toy',
    'neurips-cup': 'toy',
    'tokyo-keyboard': 'electronic',
    'cat-cup': 'cup',
    'brown-duck': 'toy',
    'lamb': 'toy',
    'duck-banana': 'toy',
    'shiba-black': 'toy',
    'pig-cup': 'cup',
    'shiba-sleep': 'toy',
    'yellow-duck': 'toy',
    'elephant': 'toy',
    'shiba-gray': 'toy',
    'dragon': 'toy'
}

myvlm_reverse_category_dict = {
    'asian_doll': 'toy',
    'boy_funko_pop': 'toy',
    'bull': 'figurine',
    'cat_statue': 'figurine',
    'ceramic_head': 'figurine',
    'chicken_bean_bag': 'toy',
    'colorful_teapot': 'tea pot',
    'dangling_child': 'toy',
    'elephant_sphere': 'figurine',
    'elephant_statue': 'figurine',
    'espresso_cup': 'cup',
    'gengar_toy': 'toy',
    'gold_pineapple': 'household object',
    'iverson_funko_pop':'toy',
    'green_doll': 'toy',
    'maeve_dog': 'pet animal',
    'minion_toy': 'toy',
    'rabbit_toy': 'toy',
    'red_chicken': 'figurine',
    'red_piggy_bank': 'piggy bank',
    'robot_toy': 'toy',
    'running_shoes': 'shoe',
    'sheep_pillow': 'pillow',
    'sheep_plush': 'toy',
    'sheep_toy': 'toy',
    'skulls_mug': 'mug',
    'small_penguin': 'toy',
    'billy_dog': 'pet animal',
    'my_cat': 'pet animal'
}


import os

def find_keyerror_in_logs():
    """
    Searches for 'KeyError:' occurrences in log files and prints filenames where found.
    """
    log_dir = "logs"
    log_files = [os.path.join(log_dir, f"rapt_12095881_{i}.out") for i in range(51)]
    with open("concept_list.txt", 'r') as f:
        lines=f.readlines()

    for log_file in log_files:
        if os.path.exists(log_file):  # Ensure the file exists before opening
            with open(log_file, "r") as f:
                for line in f:
                    if "KeyError:" in line:
                        # print(f"KeyError found in: {log_file}")
                        line_num = int(log_file.split('_')[-1].split('.')[0])
                        with open('debug.txt', 'a') as f:
                            f.write(lines[line_num])
                        break  # Stop checking further lines in the same file


if __name__ == "__main__":
    # root = 'example_database/method_mean_feat_swapped_1_ex'
    # image_paths = []
    # import os
    # import glob
    # for cat in categories:
    #     image_paths.extend(glob.glob(f'{root}/{cat}/*.jpg'))
    
    
    # for image_path in image_paths:
    #     cat, image_name = image_path.split('/')[-2:]
    #     concept = image_name.split('_')[0]
    #     json_file = f'descriptions/{cat}_{concept}.json'
    #     if not os.path.exists(json_file):
    #         print(json_file)
    # find_keyerror_in_logs()
    
    concepts = os.listdir('../MyVLM/data_myvlm/')
    for concept in concepts:
        with open('myvlm_concept_list.txt', 'a') as f:
            f.write(f'all,{concept}\n')
    
