from openai import OpenAI
import base64
from PIL import Image
import io
import re
from langchain_ollama import OllamaEmbeddings
from sklearn.decomposition import PCA
import numpy as np
import torch
import cv2
import h5py


# ---------------------- image to text sequence by llm (llama3) ----------------------
def obtain_image_code_base64(img_dir: str):
    with Image.open(img_dir) as img:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def image_file_path_2_numpy_4d(image_files: list[str]):
    result_np = None
    for idx, im_f in enumerate(image_files):
        img = cv2.imread(im_f)
        img = img[np.newaxis, :, :, :]
        if 0 == idx:
            result_np = img
        else:
            result_np = np.concatenate((result_np, img), axis=0)
    return result_np


def numpy_4d_to_base64(img_np_4d: np.ndarray): # type: ignore
    based64_list = []
    for item in img_np_4d:
        if item.dtype == np.float32 or item.dtype == np.float64:
            item = (item * 255).astype(np.uint8)
        img = Image.fromarray(item)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        based64_list.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    return based64_list


def image_file_path_2_tensor_list(image_files: list[str]):
    result_ts = []
    for im_f in image_files:
        img = cv2.imread(im_f)
        img_t = torch.from_numpy(img)
        result_ts.append(img_t.permute(2, 0, 1))
    return result_ts


def tensor_list_to_base64(tensor_list: []): # type: ignore
    based64_list = []
    for item in tensor_list:
        item = item.permute(1, 2, 0)
        img = Image.fromarray(item.numpy())
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        based64_list.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
    return based64_list


# prompt_variable = """
#     Prior Knowledge:
#         1. Img1/Img2 (MODIS, different dates): Capture temporal variation due to vegetation differences, phenological changes, and spectral signatures.
#         2. Img3 (Landsat, same date as Img2, higher resolution): Shows spatial details of geo-objects.
#         3. RGB composite (R=near-infrared, G=red, B=green): Bright red = high NIR reflectance (healthy crops); bluish areas = bare soil.
#         4. All three images cover the same area.
#         5. The area may or may not contain the following geo-objects (crop fields, rivers, bare soil, vegetation, villages, etc.), please recognise them according to the provided three images.
#         6. The bluish and greenish color indicate bare soil, the darkness tone of which may be highly related to soil moisture.
#         7. Dark blue, brown, or black may indicate the appearence of water body.
#     Questions:
#         1. According to the third image (Landsat), are there agricultural fields (crop fields)? If yes, their geometric pattern (circular/rectangular)? If no, probable geo-objects (e.g., bare soil? water body?)?
#         2. According to image 1 and 2 (Modis images of the same area but different dates), Is temporal change conspicuous?
#         3. What can you see in the image set?
#         4. For a spatiotemporal fusion model, what to prioritize using Img1-Img2's temporal difference and Img3's spatial details?        
#     Answer the questions above according to the prior knowledge and the three remote sensing images. Ensure your final response is strictly 100 words.
# """


prompt_variable = """
    Prior Knowledge:
        1. Img1/Img2 (MODIS, different dates): Capture temporal variation due to vegetation differences, phenological changes, and spectral signatures.
        2. Img3 (Landsat, same date as Img2, higher resolution): Shows spatial details of geo-objects.
        3. RGB composite (R=near-infrared, G=red, B=green): Bright red = high NIR reflectance (healthy crops); bluish areas = bare soil.
        4. All three images cover the same area.
        5. The area may or may not contain the following geo-objects (crop fields, rivers, bare soil, vegetation, villages, etc.), please recognise them according to the provided three images.
        6. The bluish and greenish color indicate bare soil, the darkness tone of which may be highly related to soil moisture.
        7. Dark blue, brown, or black may indicate the appearence of water body.
    Questions:
        According to image 1 and 2 (Modis images of the same area but different dates), Is temporal change conspicuous?
        What can you see in the image set?
        For a spatiotemporal fusion model, what to prioritize using Img1-Img2's temporal difference and Img3's spatial details?        
    Answer the questions above according to the prior knowledge and the three remote sensing images. Ensure your final response is strictly 100 words.
"""


def threesome_images_to_text_via_llm_multi_modal(
    image_tensors: [], # type: ignore
    max_token_len=200,
    temperature=0.6,
    model: str = 'gemma3:4b',
    output_text: bool = False,
    prompt: str = prompt_variable
) -> str:
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    
    image_data = tensor_list_to_base64(image_tensors)  # [obtain_image_code_base64(img) for img in image_files]
    
    # prompt = 'describe the provided three images which are remotely sensed data'
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data[0]}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data[1]}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data[2]}", "detail": "auto"}}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_token_len,
        temperature=temperature
    )
    text = response.choices[0].message.content.strip()
    if output_text:
        print(text)
    return text


# ---------------------- extract multipe tokens from original text sequence ----------------------
def text_to_tokens(text: str) -> list[str]:
    raw_tokens = re.findall(r'\b[\w-]+\b', text.lower())  # lower() lowercase    
    # filter out ineffective tokens
    valid_tokens = []
    for token in raw_tokens:
        if token.isdigit():
            continue
        if len(token) < 2:
            continue
        valid_tokens.append(token)    
    return valid_tokens


# ---------------------- Token embedding ----------------------
def tokens_to_embedding_matrix(tokens: list[str], model_name: str = "nomic-embed-text") -> np.ndarray:
    embeddings = OllamaEmbeddings(
        model=model_name,
        base_url="http://localhost:11434"
    )
    
    token_embeddings = embeddings.embed_documents(tokens)
    embedding_matrix = np.array(token_embeddings)  # (M, 4096)
    
    return embedding_matrix


# ---------------------- dimension reduction ----------------------
def reduce_token_dimension(embedding_matrix: np.ndarray, target_token_len: int = 100) -> np.ndarray:
    M, _ = embedding_matrix.shape
    if M <= target_token_len:
        # print('short text')
        pad_num = target_token_len - M
        padded_matrix = np.pad(
            embedding_matrix,
            pad_width=((0, pad_num), (0, 0)),
            mode="constant",
            constant_values=0
        )
        return padded_matrix
    else:
        reduced_matrix = embedding_matrix[0: target_token_len, :]
 
    return reduced_matrix


# ---------------------- overall process ----------------------
def full_pipeline(
    image_tensors: [],  # type: ignore
    max_token_len=120,
    temperature=0.6,
    llm_model: str = "llama3",
    embedding_model: str = 'nomic-embed-text',
    target_token_len: int = 100,
    output_text: bool = False,
) -> torch.Tensor:
    # Step1
    text = threesome_images_to_text_via_llm_multi_modal(image_tensors, max_token_len=max_token_len, temperature=temperature, model=llm_model, output_text=output_text)    
    # Step2
    embedding_matrix = tokens_to_embedding_matrix(tokens=text, model_name=embedding_model)    
    final_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    return final_tensor, text


def normalize_to_uint8(x):
    x = x.float()
    for c in range(x.size(1)):
        min_val = x[:, c, :, :].min()
        max_val = x[:, c, :, :].max()
        if max_val == min_val:
            x[:, c, :, :] = 0.0
        else:
            x[:, c, :, :] = (x[:, c, :, :] - min_val) / (max_val - min_val)
    x = (x * 255).clamp(0, 255).byte()
    return x


def normalize_tensors(t1, t2, t3, channel_index=[3, 2, 1]):
    v1 = t1[:, channel_index, :, :]
    v2 = t2[:, channel_index, :, :]
    v3 = t3[:, channel_index, :, :]    
    v1 = normalize_to_uint8(v1)
    v2 = normalize_to_uint8(v2)
    v3 = normalize_to_uint8(v3)    
    return v1, v2, v3


def ExtractTextFeaturesFromTensors(
    t1, t2, t3,
    channel_index=[3, 2, 1],
    target_token_len=100,
    max_token_len=150,
    temperature=0.6,
    llm_model="gemma3:4b",
    embedding_model='nomic-embed-text',
    output_text = False,
):
    B, _, _, _ = t1.shape
    aa, bb, cc = normalize_tensors(t1, t2, t3, channel_index)    
    text_feat_t = None    
    for idx in range(B):
        text_feat, text = full_pipeline(
            image_tensors=[aa[idx], bb[idx], cc[idx]],
            max_token_len=max_token_len,
            temperature=temperature,
            llm_model=llm_model,
            embedding_model=embedding_model,
            target_token_len=target_token_len,
            output_text = output_text
        )
        if 0 == idx:
            text_feat_t = text_feat.unsqueeze(0)
        else:
            text_feat_t = torch.cat((text_feat_t, text_feat.unsqueeze(0)), dim=0)
    return text_feat_t, text


#-------------------------------make text feature dataset by using h5 file dataset of stf task------------------------------------
def MakeTextFeatureH5FileByUsingH5Dataset(
    stf_h5_file: str = '',
    dst_h5_file: str = '',
    channel_index=[3, 2, 1],
    target_token_len=100,
    max_token_len=200,
    temperature=0.6,
    llm_model="gemma3:4b",
    embedding_model='nomic-embed-text',
    output_text = True
):
    hf = h5py.File(stf_h5_file, 'r+')
    modis_tar = np.float32(hf['modis_tar'])
    modis_ref = np.float32(hf['modis_ref'])
    landsat_ref = np.float32(hf['landsat_ref'])
    total_num = len(modis_tar)
    text_featureset = []
    for idx in range(total_num):
        m_tar, m_ref, l_ref = torch.from_numpy(modis_tar[idx]), torch.from_numpy(modis_ref[idx]), torch.from_numpy(landsat_ref[idx])
        text_f, _ = ExtractTextFeaturesFromTensors(
            m_tar.unsqueeze(0), m_ref.unsqueeze(0), l_ref.unsqueeze(0),
            channel_index, target_token_len, max_token_len, temperature, llm_model, embedding_model, output_text)
        text_featureset.append(text_f)
        print('processed %d image sets, (%d in total)' % ((idx + 1), total_num))
    text_featureset = np.stack([t.cpu().detach().numpy() for t in text_featureset])
    with h5py.File(dst_h5_file, 'w') as hf:
        hf.create_dataset('text_features', data=text_featureset, dtype=text_featureset.dtype)
        
        
#--------------------------given key words to output keyword embeddings-------------------------------------------------

def from_key_words_to_embedding(
    key_words: list[str] = [
        'modis', 'landsat', 'coarse resolution', 'fine resolution', 'spatial temporal fusion',
        'near infrared', 'nir', 'red', 'green', 'blue', 'short wave near infrared',
        'temporal', 'difference', 'change', 'variation',
        'spatial', 'spectral', 'detail', 'texture',
        'conspicuous', 'inconspicuous', 'large', 'small',
        'vegetation', 'crop', 'fallow', 'village', 'water', 'river',
    ],
    model_name='nomic-embed-text',
):
    return tokens_to_embedding_matrix(key_words, model_name=model_name)
    

# ---------------------- example ----------------------

def extract_text_feature(text_series, model_name='nomic-embed-text'):
    try:
        embeddings = OllamaEmbeddings(
            model=model_name,
            base_url="http://localhost:11434"
        )
        
        token_embeddings = embeddings.embed_documents(text_series)
        embedding_vec = np.array(token_embeddings)
        return embedding_vec
    
    except Exception as e:
        print(f"Error in text feature extraction: {str(e)}")
        return None


if '__main__' == __name__:
    # B, C, H, W = 3, 6, 100, 100
    # aa = torch.randint(low=-100, high=1000, size=(B, C, H, W))
    # bb = torch.randint(low=-100, high=1000, size=(B, C, H, W))
    # cc = torch.randint(low=-100, high=1000, size=(B, C, H, W))
    # aa = torch.from_numpy(cv2.imread('/mnt/d/docs/my paper/Multi-modal LLM-assisted deep network for spatial temporal fusion of remote sensing data/example data/l1.tif'))
    # bb = torch.from_numpy(cv2.imread('/mnt/d/docs/my paper/Multi-modal LLM-assisted deep network for spatial temporal fusion of remote sensing data/example data/m1.tif'))
    # cc = torch.from_numpy(cv2.imread('/mnt/d/docs/my paper/Multi-modal LLM-assisted deep network for spatial temporal fusion of remote sensing data/example data/m2.tif'))
    # aa = aa.permute(2, 0, 1).to(dtype=torch.int16).unsqueeze(0)
    # bb = bb.permute(2, 0, 1).to(dtype=torch.int16).unsqueeze(0)
    # cc = cc.permute(2, 0, 1).to(dtype=torch.int16).unsqueeze(0)
    # text_feat_t = ExtractTextFeaturesFromTensors(aa, bb, cc, channel_index=[0, 1, 2], llm_model='llama3', embedding_model='embeddinggemma', target_token_len=100, max_token_len=120, output_text=True)
    # print(f"final tensor shape: {text_feat_t.shape}")
    
    prompt_ablation_info = '_ablate_q1.h5'
    MakeTextFeatureH5FileByUsingH5Dataset(
        '../test_set_cia.h5',
        '../test_set_cia_text_nomic_embed_text' + prompt_ablation_info,
    )
    MakeTextFeatureH5FileByUsingH5Dataset(
        '../train_set_cia.h5',
        '../train_set_cia_text_nomic_embed_text' + prompt_ablation_info,
    )
    MakeTextFeatureH5FileByUsingH5Dataset(
        '../test_set_lgc.h5',
        '../test_set_lgc_text_nomic_embed_text' + prompt_ablation_info,
    )
    MakeTextFeatureH5FileByUsingH5Dataset(
        '../train_set_lgc.h5',
        '../train_set_lgc_text_nomic_embed_text' + prompt_ablation_info,
    )
    
