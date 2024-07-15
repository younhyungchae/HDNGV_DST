import torch
import json
import gradio as gr
import time

from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ultralytics import YOLO
from templates import *
from copy import deepcopy
from openai import OpenAI, BadRequestError

def most_common_element(lst):
    count = Counter(lst)
    most_common = count.most_common(1)
    return most_common[0][0]

def get_gpt_response(query, num_sample=1):
    # get response from gpt-3.5-turbo
    try:
        response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": query},
                ],
                n=num_sample
                #response_format={ "type": "json_object" }
            )
    except BadRequestError:
        return 'bad_request'
    except Exception as e:
        print(e)
        raise
    result = [item.message.content for item in response.choices]
    return result

def get_response(image, query):
    # get single response from multimodal large language model
    inp = query
    image_processor = processor['image']
    conv_mode = "phi"  # qwen or stablelm
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    return outputs

def get_classification_result(query, candidates):
    # do classification with large language model
    inputs = llm_tokenizer(query, return_tensors="pt").to(llm_model.device)
    outputs = llm_model(**inputs)
    logits = outputs.logits[0][-1]
    candidate_ids = [llm_tokenizer.convert_tokens_to_ids(candidate) for candidate in candidates]
    return candidates[torch.argmax(torch.stack([logits[candidate_id] for candidate_id in candidate_ids]))]

def get_llm_response(query):
    # get large language model response
    inputs = llm_tokenizer(query, return_tensors="pt").to(llm_model.device)

    terminators = [
        llm_tokenizer.eos_token_id,
        llm_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        198,
        271
    ]

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=llm_tokenizer.eos_token_id,
        eos_token_id=terminators,
        do_sample=False,
    )
    response = outputs[0][inputs['input_ids'].shape[-1]:]
    return llm_tokenizer.decode(response, skip_special_tokens=True).strip()

def get_multi_sampling_result(template, query):
    # run llm 3 times and majority vote
    outputs = [get_llm_response(template.substitute(utterance=query)) for _ in range(3)]
    return most_common_element(outputs)

def get_bbox(img, query):
    # get bbox coordinates from referring expression
    bbox = eval(get_response(img, object_detecting_prompt.substitute(expression=query)))
    width, height = img.size
    bbox[0] *= width
    bbox[1] *= height
    bbox[2] *= width
    bbox[3] *= height
    return bbox

def get_class(img, bbox):
    # classify the class of the car of img in bounding box
    cropped_img = img.crop(bbox)
    output = yolo(cropped_img)[0]
    return output.names[output.probs.top1]

def get_intent(query):
    # classify user intent among 'TakePhoto', 'CarExplain', 'AskConfirm', 'CarCompare', 'AskInfo'
    labels = ['A','B','C','D','E']
    intents = ['TakePhoto', 'CarExplain', 'AskConfirm', 'CarCompare', 'AskInfo']
    prompt = intent_classification_prompt.substitute(utterance=query)
    intent = get_classification_result(prompt, labels)
    return intents[labels.index(intent)]

def get_slots(image, intent, query):
    # fill slots w.r.t. the user intent
    if intent == 'TakePhoto':
        slots = {'photo_spot':None, 'refer_obj':None}
        slots['photo_spot'] = get_multi_sampling_result(photo_spot_prompt, query)
        slots['refer_obj'] = get_multi_sampling_result(referring_object_prompt, query)
        if not slots['refer_obj'] == 'None':
            bbox = get_bbox(image, slots['refer_obj'])
            slots['bbox'] = bbox
            slots['car_name'] = get_class(image, bbox)

    elif intent == 'CarExplain':
        slots = {'car_name':None, 'info_type':None, 'refer_obj':None}
        slots['car_name'] = get_multi_sampling_result(explain_car_name_prompt, query)
        slots['info_type'] = get_multi_sampling_result(explain_info_type_prompt, query)
        slots['refer_obj'] = get_multi_sampling_result(referring_object_prompt, query)
        if not slots['refer_obj'] == 'None':
            bbox = get_bbox(image, slots['refer_obj'])
            slots['bbox'] = bbox
            slots['car_name'] = get_class(image, bbox)

    elif intent == 'AskConfirm':
        slots = {'car_name': None, 'info_type': None, 'refer_obj':None}
        slots['car_name'] = get_multi_sampling_result(confirm_car_name_prompt, query)
        slots['info_type'] = get_multi_sampling_result(confirm_info_prompt, query)
        slots['refer_obj'] = get_multi_sampling_result(referring_object_prompt, query)
        if not slots['refer_obj'] == 'None':
            bbox = get_bbox(image, slots['refer_obj'])
            slots['bbox'] = bbox
            slots['car_name'] = get_class(image, bbox)

    elif intent == 'CarCompare':
        slots = {'target': None, 'info': None, 'refer_obj':None}
        slots['target'] = get_multi_sampling_result(compare_target_prompt, query).split(',')
        slots['info'] = get_multi_sampling_result(compare_info_prompt, query)
        slots['refer_obj'] = get_multi_sampling_result(referring_object_prompt, query)
        if not slots['refer_obj'] == 'None':
            bbox = get_bbox(image, slots['refer_obj'])
            slots['bbox'] = bbox
            slots['car_name'] = get_class(image, bbox)

    elif intent == 'AskInfo':
        slots = {'info': None}
        slots['info'] = get_multi_sampling_result(ask_info_prompt, query)
        
    else:
        raise ValueError('Wrong Intent')
    return slots

def draw_bbox(img, bboxes):
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        draw.rectangle(bbox.tolist(), outline=(0,255,0), width = 3)
    return img

# load GPT for chatbot system (optional)
client = OpenAI(
    api_key='<your-api-key>')

# load yolo as image classifier
yolo = YOLO('./runs/classify/train01/weights/best.pt')  # load a pretrained model (recommended for training)

# load multimodal large language model
disable_torch_init()
model_path = 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384'  # LanguageBind/MoE-LLaVA-Qwen-1.8B-4e or LanguageBind/MoE-LLaVA-StableLM-1.6B-4e
device = 'cuda'
load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)

# load large language model
model_id = "meta-llama/Meta-Llama-3-8B"

llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
llm_model.generation_config.temperature=None
llm_model.generation_config.top_p=None

# load processed data
with open('./data/label.json', 'r') as f:
    full_data = json.load(f)

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

# run gradio demo
with gr.Blocks() as demo:
    history = gr.State([])
    
    def respond(image, query, history):
        current_image = deepcopy(image)
        if intent.value == "":
            intent_value = get_intent(query)
            slot_values = get_slots(image, intent_value, query)
            slot_list = '\n'.join([f'{key}: {value}\n' for key, value in slot_values.items()])
            
        else:
            intent_value = intent.value
            slot_values = get_slots(image, intent_value, query)
            slot_list = '\n'.join([f'{key}: {value}\n' for key, value in slot_values.items()])
        if 'bbox' in list(slot_values.keys()):
            
            draw=ImageDraw.Draw(current_image)
            draw.rectangle(slot_values['bbox'], width=4, outline=(255,0,0))
        
        gpt_respond = get_gpt_response(gpt_prompt.substitute(intent=intent_value, slots=json.dumps(slot_values), utterance=query))
        history.append((query, gpt_respond[0]))
        time.sleep(2)
        return current_image, "", history, intent_value, slot_list
    
    def clear_everything():
        return None, '', [], '', '', []

    with gr.Row() as row:
        with gr.Column():
            image = gr.Image(type='pil')
            msg = gr.Textbox(label='Utterance', placeholder='Write new utterance')
            clear = gr.ClearButton()
        with gr.Column():
            chatbot = gr.Chatbot(label='Dialog History')
        with gr.Column():
            intent = gr.TextArea(label='intent', lines=1)
            slots = gr.TextArea(label='slots', lines=8)
    
    msg.submit(respond, [image, msg, history], [image, msg, chatbot, intent, slots])
    clear.click(clear_everything, [], [image, msg, chatbot, intent, slots, history])
    
demo.launch(share=True)


