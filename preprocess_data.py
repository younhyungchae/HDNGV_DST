import os
import json

image_path = './multimodal_data_v2/data/'
label_path = os.path.join(image_path, 'label')
data_list = os.listdir(label_path)

full_data = []
for path in data_list:
    try:
        with open(os.path.join(label_path, path), 'r') as f:
            data = json.load(f)
        full_data.append({'image':path.replace('json','jpg'), 'data':data})
    except Exception as e:
        print(e)
        print(path)

for data_idx, data in enumerate(full_data):
    for dialogue_idx, dialogue in enumerate(data['data']):
        for turn_idx, turn in enumerate(dialogue):
            if turn['intent'] == '<ConfirmInfo>':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'ConfirmInfo'
            elif turn['intent'] == 'takePhoto':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'TakePhoto'
            elif turn['intent'] == '<CarExplain>':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'CarExplain'
            elif turn['intent'] == 'carExplain':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'CarExplain'
            elif turn['intent'] == 'takePhoto_together':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'TakePhoto'
            elif turn['intent'] == 'takePhoto_alone':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'TakePhoto'
            elif turn['intent'] == 'convenience':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'CarExplain'
            elif turn['intent'] == 'carCompare':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = 'CarCompare'

for data_idx, data in enumerate(full_data):
    for dialogue_idx, dialogue in enumerate(data['data']):
        for turn_idx, turn in enumerate(dialogue):
            if turn['intent']=='TakePhoto':
                if 'carInfoType' in turn['slot'].keys():
                    full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot'].pop('carInfoType')
                if 'carName' in turn['slot'].keys():
                    full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot'].pop('carName')

from copy import deepcopy

for data_idx, data in enumerate(full_data):
    for dialogue_idx, dialogue in enumerate(data['data']):
        for turn_idx, turn in enumerate(dialogue):
            if turn['intent']=='CarExplain' and 'carname' in turn['slot'].keys():
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['carName'] = deepcopy(turn['slot']['carname'])
                del full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['carname']

for data_idx, data in enumerate(full_data):
    for dialogue_idx, dialogue in enumerate(data['data']):
        for turn_idx, turn in enumerate(dialogue):
            if turn['intent']=='CarExplain' and '<TargetInfo>' in turn['slot'].keys():
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['carInfoType'] = deepcopy(turn['slot']['<TargetInfo>'])
                del full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['<TargetInfo>']
                
            if turn['intent']=='ConfirmInfo' and '<TargetInfo>' in turn['slot'].keys():
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['carInfoType'] = deepcopy(turn['slot']['<TargetInfo>'])
                del full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['<TargetInfo>']
                
            if turn['intent']=='TakePhoto' and 'photo_spot' in turn['slot'].keys():
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['Position'] = deepcopy(turn['slot']['photo_spot'])
                del full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot']['photo_spot']

for data_idx, data in enumerate(full_data):
    for dialogue_idx, dialogue in enumerate(data['data']):
        for turn_idx, turn in enumerate(dialogue):
            if turn['intent']=='ConfirmInfo':
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['intent'] = full_data[data_idx]['data'][dialogue_idx][turn_idx-1]['intent']
                full_data[data_idx]['data'][dialogue_idx][turn_idx]['slot'] = full_data[data_idx]['data'][dialogue_idx][turn_idx-1]['slot']

slot_info = {}
for data in full_data:
    for dialogue in data['data']:
        for turn in dialogue:
            if turn['intent'] in slot_info.keys():
                for slot in turn['slot'].keys():
                    if not slot in slot_info[turn['intent']]:
                        slot_info[turn['intent']].append(slot)
            else:
                slot_info[turn['intent']] = list(turn['slot'].keys())

slot_info

with open('./multimodal_data_v2/data/label.json','w') as f:
    json.dump(full_data,f)