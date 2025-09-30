import os
import json

def extract_third_level_tags(input_folder, output_folder):
    cnt = 0
    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue
            
        input_json_path = os.path.join(input_folder, filename)
        
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        for scene in data['scenes']:
            for second_level_scene in scene['sub_tag']:
                for third_level_scene in second_level_scene['sub_tag']:
                    data_file = data['data_file'].replace('.mp4', '')
                    new_json_data = {
                        "date": data['date'],
                        "data_file": data_file,
                        "scene_tag": scene['tag'],
                        "sub_scene_tag": second_level_scene['tag'],
                        "third_level_tag": third_level_scene['tag'],
                        "start_time": third_level_scene['start_time'],
                        "end_time": third_level_scene['end_time'],
                        "duration": third_level_scene['duration'],
                        "instance_id": third_level_scene.get('instance_id', [])
                    }
                    
                    duration = third_level_scene['duration']
                    if duration <= 3:
                        continue
                    
                    cnt += 1
                    new_json_filename = f"{data_file}_{new_json_data['start_time']}_{new_json_data['end_time']}.json"
                    new_json_path = os.path.join(output_folder, new_json_filename)

                    with open(new_json_path, 'w') as new_f:
                        json.dump(new_json_data, new_f, indent=4)

    print(f"Total segments extracted from {input_json_path}: {cnt}")

input_folder = '/lab/haoq_lab/12532563/xpool/data/suscape/labels'
output_folder = '/lab/haoq_lab/12532563/xpool/data/suscape/third_level_labels'

os.makedirs(output_folder, exist_ok=True)

extract_third_level_tags(input_folder, output_folder)
