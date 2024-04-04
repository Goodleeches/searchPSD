import tkinter as tk
import os
import cv2
import numpy as np
import sys
#import threading
from tkinter import filedialog
from psd_tools import PSDImage
from PIL import Image, ImageTk
from tkinter import ttk
#from threading import Thread
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager
import time
from multiprocessing import Pool, Manager, Lock
import os
import logging
import threading
from multiprocessing import current_process
import psutil
from tkinter import messagebox
import subprocess
import copy

def update_log_messages(log_queue):
    while not log_queue.empty():
        message = log_queue.get()
        if len(message) > 30:
            split_point = len(message) // 2
            nearest_space = message.rfind(' ', 0, split_point)
            if nearest_space != -1:
                message = message[:nearest_space] + '\n' + message[nearest_space+1:]
            else:
                message = message[:split_point] + '\n' + message[split_point:]
        label_status.config(text=message)
        tkRoot.update_idletasks()
        print("log_update:" + message);
    #log_message("log update!!");
    tkRoot.after(1000, lambda: update_log_messages(log_queue)) 

def log_message(*messageArgs, log_queue=None):
    process_name = current_process().name  # 현재 프로세스의 이름을 얻습니다.
    message = " ".join(str(arg) for arg in messageArgs)
    full_message = f"[{process_name}] {message}"  # 프로세스 이름을 메시지 앞에 추가합니다.
    print(full_message)
    #log_queue.put(message)
    #time.sleep(1)

def show_selected_image(image):
    """ 선택된 이미지를 Label 위젯에 표시하는 함수 """
    try:
        # 이미지 로드 및 Tkinter 포맷으로 변환
        pil_image = image
        pil_image = pil_image.resize((250, 250))  # 이미지 크기 조정
        tk_image = ImageTk.PhotoImage(pil_image)

        # Label 위젯에 이미지 표시
        image_label.config(image=tk_image)
        image_label.image = tk_image  # 참조를 유지하기 위해 이렇게 설정
    except Exception as e:
        log_message(f"Error displaying image: {e}")
        
def on_image_selected(event, args):
    log_queue, processed_files, lock = args
    """ 사용자가 이미지를 선택했을 때 실행되는 함수 """
    selected_index = listbox_image_details.curselection()
    #log_message(f"selected_index: {selected_index}", )
    
    if not selected_index:
        return

    selected_text = listbox_image_details.get(selected_index)
    selected_image_name = selected_text.split("  -  ")[0]
    global merged_dict
    if last_path_select in merged_dict and selected_image_name in merged_dict[last_path_select]:
        image = merged_dict[last_path_select][selected_image_name]["image"]
        show_selected_image(image)


def show_image_details(event, args):
    log_queue, processed_files, lock = args
    """ 선택된 경로에 있는 이미지들의 상세 정보를 표시하는 함수 """
    global merged_dict
    
    selected_index = listbox_processed_files.curselection()
    if not selected_index:
        return

    selected_text = listbox_processed_files.get(selected_index[0])
    selected_path = selected_text.split("  -  ")[0]
    global last_path_select
    last_path_select = selected_path

    listbox_image_details.delete(0, tk.END)

    if not merged_dict:
        return;

    if selected_path in merged_dict:
        # 유사도에 따라 정렬
        sorted_images = sorted(merged_dict[selected_path].items(), key=lambda x: x[1]['similarity'], reverse=True)
        for image_name, image_info in sorted_images:
            listbox_image_details.insert(tk.END, f"{image_name}  -  Similarity: {image_info['similarity']:.2f}")

def show_processed_files(args, merged_dict):
    log_queue, processed_files, lock = args
    
    if not merged_dict:
        log_message(f"processed_files Size: empty", log_queue=log_queue)
        return
    
    highest_similarity_per_path = []
    for path, images in merged_dict.items():
        highest_similarity = max(images.items(), key=lambda x: x[1]["similarity"])
        highest_similarity_per_path.append((path, *highest_similarity))

    # 유사도가 높은 순으로 정렬
    sorted_by_similarity = sorted(highest_similarity_per_path, key=lambda x: x[2]["similarity"], reverse=True)

    # Listbox에 한 번에 추가할 아이템 목록 생성
    listbox_items = [f"{path}  -  Similarity: {image_info['similarity']:.2f}" 
                     for path, image_name, image_info in sorted_by_similarity]

    # 기존 Listbox 아이템을 모두 삭제하고 새 목록을 추가
    log_message(f"listbox_processed_files.delete")
    listbox_processed_files.delete(0, tk.END)
     # 예외 처리 추가
    try:
        log_message(f"listbox_processed_files.insert s")
        listbox_processed_files.insert(tk.END, *listbox_items)
        log_message(f"listbox_processed_files.insert e")
    except Exception as e:
        log_message(f"Error while inserting into listbox: {e}", log_queue=log_queue)
        # 사용자에게 오류 메시지를 표시하거나 다른 처리를 수행할 수 있는 부분
        
def safe_path(path):
    """ 파일 경로에서 비-ASCII 문자를 그대로 유지하고, 파일 존재 여부를 확인하는 함수 """
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(path):
            log_message(f"File does not exist: {path}", log_queue=log_queue)
            return None
        return path
    except Exception as e:
        log_message(f"Error processing path: {path}, Error: {e}", log_queue=log_queue)
        return None

def merge_group_layers(psd, group, base_path, group_name=''):
    """빈 캔버스에 레이어 별로 이미지를 붙여넣어 그룹 레이어 또는 단일 레이어를 병합합니다."""
    # 초기 캔버스 크기를 설정합니다.
    max_width = psd.width
    max_height = psd.height
    
    # 단일 레이어 처리를 위한 분기
    if not hasattr(group, '__iter__'):
        layers = [group]  # 단일 레이어를 리스트로 만듭니다.
    else:
        layers = group  # 그룹 레이어의 경우, 바로 사용합니다.

    # for layer in layers:
    #     if not layer.visible:
    #         continue
        # layer.bbox는 (left, top, right, bottom)을 반환합니다.
        #max_width = max(max_width, layer.bbox[2])  # layer.bbox[2]는 레이어의 right 값
        #max_height = max(max_height, layer.bbox[3])  # layer.bbox[3]는 레이어의 bottom 값
    
    # 충분히 큰 캔버스 생성
    merged_image = Image.new('RGBA', (max_width, max_height))

    def process_layer(layer):
        if not layer.visible:
            return

        if hasattr(layer, 'is_group') and layer.is_group():
            for sub_layer in layer:
                process_layer(sub_layer)
        else:
            layer_image = layer.topil().convert('RGBA') if layer.topil() else None
            if layer_image:
                # x_offset = parent_bbox[0] + layer.bbox[0]
                # y_offset = parent_bbox[1] + layer.bbox[1]
                pos_x = layer.bbox[0]
                pos_y = layer.bbox[1]
                merged_image.paste(layer_image, (pos_x, pos_y), layer_image)

    for layer in layers:
        process_layer(layer)

    if merged_image:
        merged_image_name = f"{base_path}_Merged.png"
        #merged_image_path = os.path.join(os.path.dirname(base_path), merged_image_name)
        #merged_image.save(merged_image_name)
        return merged_image

    return None


def process_psd_file(psd_file_path, compare_with, args):
    log_queue, processed_files, lock = args
    """ PSD 파일을 처리하는 함수 """
    safe_psd_file_path = safe_path(psd_file_path)
    if not safe_psd_file_path:
        log_message(f"Error processing PSD file path: {psd_file_path}")
        return

    ret = {}
    mergered_img_list = []
    psd = PSDImage.open(safe_psd_file_path)
    for layer in psd:
        if not layer.visible:
            continue
        merged_image = merge_group_layers(psd, layer, safe_psd_file_path, compare_with)
        mergered_img_list.append(merged_image)
        if merged_image:
            log_message("Merge")
            merged_image_name = f"{layer.name}.png"
            #merged_image.save(merged_image_name)
            result = compare_images_in_memory(compare_with, merged_image, safe_psd_file_path, merged_image_name, processed_files, lock)
            for path, data in result.items():
                if path in ret:
                    ret[path].update(data)
                else:
                     ret[path] = data
    
    max_width = psd.width
    max_height = psd.height
    empty_image = Image.new('RGBA', (max_width, max_height))        
    for merged in reversed(mergered_img_list):
        empty_image = Image.alpha_composite(merged, empty_image)
        
    log_message("Merged Merge")
    merged_image_name_Merged = f"{layer.name}_Merged.png"
    #empty_image.save(merged_image_name_Last)
    compare_images_in_memory(compare_with, empty_image, safe_psd_file_path, merged_image_name_Merged, processed_files, lock)     
    merged_image_name_Merged_Template = f"{layer.name}_Search_In_Merged_Template_Image.png"  
    result = find_template_in_image(compare_with, empty_image, safe_psd_file_path, merged_image_name_Merged_Template, args);
    for path, data in result.items():
        if path in ret:
            ret[path].update(data)
        else:
            ret[path] = data
    return ret
    
def process_psd_preview(psd_file_path, compare_with, args):
    log_queue, processed_files, lock = args
    """ PSD 파일의 미리보기 이미지를 처리하는 함수 """
    safe_psd_file_path = safe_path(psd_file_path)
    if not safe_psd_file_path:
        log_message(f"Error processing PSD file path: {psd_file_path}", log_queue=log_queue)
        return

    try:
        psd = PSDImage.open(safe_psd_file_path)
        # 미리보기 이미지 추출
        preview_image = psd.composite()

        # 미리보기 이미지 저장 (옵션)
        # preview_image_name = f"{safe_psd_file_path}_Preview.png"
        # preview_image.save(preview_image_name)

        # 미리보기 이미지와 비교 이미지를 메모리 상에서 비교
        return compare_images_in_memory(compare_with, preview_image, safe_psd_file_path, "Preview_Image", processed_files, lock)

    except Exception as e:
        log_message(f"Failed to process PSD file: {safe_psd_file_path}. Error: {e}", log_queue=log_queue)
    
    
def select_file():
    """ Let the user select a PNG file. """
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    file_path = os.path.normpath(file_path)
    if file_path:
        global selected_png
        selected_png = file_path
        message = f"Selected PNG File: {file_path}"
        if len(message) > 50:
            split_point = len(message) // 2
            nearest_space = message.rfind(' ', 0, split_point)
            if nearest_space != -1:
                message = message[:nearest_space] + '\n' + message[nearest_space+1:]
            else:
                message = message[:split_point] + '\n' + message[split_point:]
        label_selected_file.config(text=message)
        
        # 선택한 이미지를 우측 하단에 표시
        try:
            # 이미지 로드 및 Tkinter 포맷으로 변환
            pil_image = Image.open(file_path)  # PIL로 이미지를 로드
            pil_image = pil_image.resize((250, 250))  # 이미지 크기 조정
            tk_image = ImageTk.PhotoImage(pil_image)

            # 이미지를 표시할 Label 위젯에 이미지 표시
            select_image_label.config(image=tk_image)
            select_image_label.image = tk_image  # 참조를 유지하기 위해 이렇게 설정
        except Exception as e:
            log_message(f"Error displaying selected PNG image: {e}")
    else:
        label_selected_file.config(text="No file selected.")

def select_folder():
    """ Let the user select a folder. """
    folder_path = filedialog.askdirectory()
    folder_path = os.path.normpath(folder_path)
    if folder_path:
        global selected_folder
        selected_folder = folder_path
        message = f"Selected Folder: {folder_path}"
        if len(message) > 50:
            split_point = len(message) // 2
            nearest_space = message.rfind(' ', 0, split_point)
            if nearest_space != -1:
                message = message[:nearest_space] + '\n' + message[nearest_space+1:]
            else:
                message = message[:split_point] + '\n' + message[split_point:]
        label_selected_folder.config(text=message)    
    else:
        label_selected_folder.config(text="No folder selected.")

def imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    try:
        # 파일을 바이트로 읽고 numpy 배열로 변환
        data = np.fromfile(filename, dtype=np.uint8)
        # cv2.imdecode 함수를 사용하여 이미지 로드
        return cv2.imdecode(data, flags)
    except Exception as e:
        log_message(e)
        return None

# def update_dict(processed_files, lock, path, imageName, image, similarity):
#     with lock:
#         if path not in processed_files:
#             log_message("processed_files size: " + str(len(processed_files)));
#             processed_files[path] = {}
#             log_message("processed_files size: " + str(len(processed_files)));

#         if imageName not in processed_files[path]:
#             temp = processed_files[path]
#             temp[imageName] = {"similarity": similarity, "image": image}
#             #log_message(temp[imageName])
#             #log_message(temp)
#             processed_files[path] = temp  # 변경 사항을 명시적으로 동기화
#             log_message("==================================");
#             log_message("update_dict path: " + str(path));
#             log_message("update_dict imageName: " + str(imageName));
#             log_message("update_dict image: " + str(image));
#             log_message("update_dict similarity: " + str(similarity));
#             log_message(processed_files[path])
#             log_message("==================================");      
            
def update_dict(processed_files, lock, path, imageName, image, similarity):
    temp_dict = {};
    temp_dict[path] = {}
    temp_dict[path][imageName] = {"similarity": similarity, "image": image}
    log_message(temp_dict)
    return temp_dict  # 수정된 딕셔너리 반환
    #local_processed_files[path] = temp  # 변경 사항을 명시적으로 동기화
    log_message("==================================");
    log_message("update_dict path: " + str(path));
    log_message("update_dict imageName: " + str(imageName));
    log_message("update_dict image: " + str(image));
    log_message("update_dict similarity: " + str(similarity));
    log_message(temp)
    log_message("==================================");    
    
    # temp = processed_files.get(path, {})  # Lock 없이 접근
    # with lock:  # Lock은 최소한의 범위에서만 사용
    #     log_message("!!! LOCK !!!");
    #     temp[imageName] = {"similarity": similarity, "image": image}
    #     processed_files[path] = temp  # 변경 사항을 명시적으로 동기화
    # log_message("==================================");
    # log_message("update_dict path: " + str(path));
    # log_message("update_dict imageName: " + str(imageName));
    # log_message("update_dict image: " + str(image));
    # log_message("update_dict similarity: " + str(similarity));
    # log_message(temp)
    # log_message("==================================");      
                  
    
def update_dict_if_higher(my_dict, key, new_value):
    if key not in my_dict or my_dict[key] < new_value:
        my_dict[key] = new_value    

def find_template_in_image(image_path, mergedImage, PsdPath, NamePath, args):
    log_queue, processed_files, lock = args
    # 원본 이미지를 로드합니다.
    main_image = imread_unicode(image_path)
    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

    # 템플릿 이미지를 그레이스케일로 변환합니다.
    template_gray = cv2.cvtColor(np.array(mergedImage), cv2.COLOR_RGB2GRAY)

    if main_image_gray.shape[0] < template_gray.shape[0] or main_image_gray.shape[1] < template_gray.shape[1]:
        # 템플릿 크기에 맞춰 main_image의 새로운 크기 계산
        # 여기서는 예시로 템플릿 크기와 동일하게 조정합니다.
        # 실제로는 비율을 유지하면서 조정할 수도 있습니다.
        new_width = template_gray.shape[1]
        new_height = template_gray.shape[0]
        main_image_gray = cv2.resize(main_image_gray, (new_width, new_height))

    # 템플릿 매칭을 수행합니다.
    res = cv2.matchTemplate(main_image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 최대 매칭 포인트를 찾습니다.
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 최대 매칭 포인트에 사각형을 그립니다.
    top_left = max_loc
    h, w = template_gray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(main_image, top_left, bottom_right, (255, 0, 0), 2)

    # 매칭 결과 이미지를 표시합니다.
    # cv2.imshow('Template Found', main_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    similarity = max_val;
    #listbox_processed_files.insert(tk.END, f"{image2_path} - Last Similarity: {similarity:.2f}")
    return update_dict(processed_files, lock, PsdPath, NamePath, mergedImage, similarity);

def check_and_load_image(image_path):
    # 파일 경로와 존재 여부 확인
    if not os.path.exists(image_path):
        log_message(f"File does not exist: {image_path}")
        return None

    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        log_message(f"Failed to read image: {image_path}")
    return image

def check_and_load_image_pil(image_path):
    """ 파일 경로를 확인하고 PIL을 사용하여 이미지를 읽는 함수 """
    try:
        with Image.open(image_path) as img:
            original_img = img.copy()  # 원본 이미지 복사
            grayscale_img = img.convert('L')  # 'L' 모드로 그레이스케일 변환
            return original_img, grayscale_img
    except IOError:
        log_message(f"Failed to open image: {image_path}")
        return None
    
def resize_image(image, size):
    """이미지를 주어진 크기로 리사이징하는 함수"""
    return image.resize(size, Image.Resampling.LANCZOS)

def calculate_ssim(image1, image2):
    # 이미지의 평균값 계산
    avg1 = np.mean(image1)
    avg2 = np.mean(image2)

    # 이미지의 분산 계산
    var1 = np.var(image1)
    var2 = np.var(image2)

    # 이미지의 공분산 계산
    covar12 = np.cov(image1.flatten(), image2.flatten())[0, 1]

    # SSIM 계산에 사용되는 상수 정의
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # SSIM 계산
    numerator = (2 * avg1 * avg2 + c1) * (2 * covar12 + c2)
    denominator = (avg1 ** 2 + avg2 ** 2 + c1) * (var1 + var2 + c2)
    ssim = numerator / denominator

    # SSIM 값이 음수인 경우 0으로 설정
    if ssim < 0:
        ssim = 0

    return ssim

def compare_images_in_memory(image1_path, image2, PsdPath, NamePath, processed_files, lock):
    """PIL과 scikit-image를 사용하여 두 이미지의 유사도를 비교하는 함수"""
    image1_origin, image1_gray = check_and_load_image_pil(image1_path)
    
    if image1_origin is None or image1_gray is None or image2 is None:
        log_message(f"Error reading one of the images: {image1_path} or {image2}")
        return
    
    none_convert = image2;
    image2 = image2.convert('L')

    # 두 이미지를 동일한 크기로 리사이징
    target_size = (100, 100)  # 예시로 100x100 크기를 사용
    log_message(f"image1 {image1_gray.width} and {image1_gray.height}")
    log_message(f"image2 {image2.width} and {image2.height}")
    image1_resized = resize_image(image1_gray, target_size)
    image2_resized = resize_image(image2, target_size)
    log_message(f"image1_resized {image1_resized.width} and {image1_resized.height}")
    log_message(f"image2_resized {image2_resized.width} and {image2_resized.height}")
    # 이미지를 NumPy 배열로 변환
    image1_np = np.array(image1_resized)
    image2_np = np.array(image2_resized)

    # 유사도 계산
    similarity = calculate_ssim(image1_np, image2_np)
    log_message(f"Similarity between {image1_path} and {PsdPath}: {similarity}")
    return update_dict(processed_files, lock, PsdPath, NamePath, none_convert, similarity);

def compare_images_pil(image1_path, image2_path, ImageName, processed_files, lock):
    """ PIL과 scikit-image를 사용하여 두 이미지의 유사도를 비교하는 함수 """
    image1_origin, image1_gray = check_and_load_image_pil(image1_path)
    image2_origin, image2_gray = check_and_load_image_pil(image2_path)
    
    if image1_origin is None or image1_gray is None or image2_origin is None or image2_gray is None:
        log_message(f"Error reading one of the images: {image1_path} or {image2_path}")
        return
    
    # 두 이미지를 동일한 크기로 리사이징
    target_size = (100, 100)  # 예시로 100x100 크기를 사용
    image1_resized = resize_image(image1_gray, target_size)
    image2_resized = resize_image(image2_gray, target_size)

    # 이미지를 NumPy 배열로 변환
    image1_np = np.array(image1_resized)
    image2_np = np.array(image2_resized)

    # 유사도 계산
    similarity = calculate_ssim(image1_np, image2_np)
    log_message(f"Similarity between {image1_path} and {image2_path}: {similarity}")
    return update_dict(processed_files, lock, image2_path, ImageName, image2_origin, similarity);
    
def process_file(png_file, file_path, log_queue, processed_files, lock):
    args = (log_queue, processed_files, lock)
    if file_path.endswith(".psd"):
        log_message(file_path, log_queue=log_queue)       
        return process_psd_file(file_path, png_file, args)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        log_message(file_path, log_queue=log_queue)       
        return compare_images_pil(png_file, file_path, os.path.basename(file_path), processed_files, lock)
    #log_message("process_file End", log_queue=log_queue)       

def process_file_wrapper(png_file, file_path, log_queue, processed_files, lock):
    #png_file, file_path, log_queue, processed_files, lock = args_tuple
    return process_file(png_file, file_path, log_queue, processed_files, lock)
    
def process_folder_parallel(png_file, folder_path, on_all_threads_completed, args):
    global pool  # 전역 변수로 pool을 선언하여 어디서든 접근 가능하게 합니다.
    local_processed_files = {}
    availableProcess = os.cpu_count()
    log_queue, processed_files, lock = args

    file_paths = [(png_file, os.path.join(root, filename), log_queue, processed_files, lock) 
                  for root, _, files in os.walk(folder_path) 
                  for filename in files if filename.endswith((".psd", ".png", ".jpg", ".jpeg"))]

    pool = Pool(processes=availableProcess)

    with Pool(processes=availableProcess) as pool:
        results = pool.starmap_async(process_file_wrapper, file_paths)
        local_processed_files_list = results.get()  # 모든 작업이 완료될 때까지 기다린 후 결과 수집
    log_message("local_processed_files_list = results")
    # 결과 병합
    # 빈 딕셔너리를 생성합니다.
    global merged_dict
    merged_dict = {}

    # 리스트 안의 모든 딕셔너리를 순회합니다.
    for local_dict in local_processed_files_list:
        # 각 딕셔너리의 키-값 쌍을 merged_dict에 추가합니다.
        for key, value in local_dict.items():
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                # 이미 키가 존재하는 경우, 더 복잡한 로직으로 처리가 필요할 수 있습니다.
                # 예를 들어, 값들을 합치거나, 리스트로 관리하는 등의 방법이 있습니다.
                # 여기서는 단순히 값을 업데이트하는 방식을 사용합니다.
                merged_dict[key].update(value)
    log_message("Merge dict")
    on_all_threads_completed(args, merged_dict)
    
def fast_process_file(png_file, file_path, log_queue, processed_files, lock):
    args = (log_queue, processed_files, lock)
    if file_path.endswith(".psd"):
        log_message(file_path, log_queue=log_queue)       
        return process_psd_preview(file_path, png_file, args)
    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        log_message(file_path, log_queue=log_queue)       
        return compare_images_pil(png_file, file_path, os.path.basename(file_path), processed_files, lock)
    #log_message("process_file End", log_queue=log_queue)  

def fast_process_file_wrapper(png_file, file_path, log_queue, processed_files, lock):
    #png_file, file_path, log_queue, processed_files, lock = args_tuple
    return fast_process_file(png_file, file_path, log_queue, processed_files, lock)
    
def fast_process_folder_parallel(png_file, folder_path, on_all_threads_completed, args):
    local_processed_files = {}
    global pool  # 전역 변수로 pool을 선언하여 어디서든 접근 가능하게 합니다.
    availableProcess = os.cpu_count()
    log_queue, processed_files, lock = args

    file_paths = [(png_file, os.path.join(root, filename), log_queue, processed_files, lock) 
                  for root, _, files in os.walk(folder_path) 
                  for filename in files if filename.endswith((".psd", ".png", ".jpg", ".jpeg"))]

    # pool 생성
    pool = Pool(processes=availableProcess)

    with Pool(processes=availableProcess) as pool:
        results = pool.starmap_async(fast_process_file_wrapper, file_paths)
        local_processed_files_list = results.get()  # 모든 작업이 완료될 때까지 기다린 후 결과 수집

    # 결과 병합
    # 빈 딕셔너리를 생성합니다.
    global merged_dict
    merged_dict = {}

    # 리스트 안의 모든 딕셔너리를 순회합니다.
    for local_dict in local_processed_files_list:
        # 각 딕셔너리의 키-값 쌍을 merged_dict에 추가합니다.
        for key, value in local_dict.items():
            if key not in merged_dict:
                merged_dict[key] = value
            else:
                # 이미 키가 존재하는 경우, 더 복잡한 로직으로 처리가 필요할 수 있습니다.
                # 예를 들어, 값들을 합치거나, 리스트로 관리하는 등의 방법이 있습니다.
                # 여기서는 단순히 값을 업데이트하는 방식을 사용합니다.
                merged_dict[key].update(value)

    # 병합된 딕셔너리를 processed_files에 할당합니다.
    #processed_files = merged_dict
    # 병합된 결과를 사용하여 추가 작업 수행
    on_all_threads_completed(args, merged_dict)
    
def on_all_threads_completed(args, merged_dict):
    log_queue, processed_files, lock = args
    #log_message("show_processed_files.", log_queue=log_queue)
    show_processed_files(args, merged_dict)
    label_status.config(text=f"completed")
    tkRoot.update_idletasks()
    progress_bar.stop()  # 프로그레스 바 정지
    progress_bar.pack_forget()  # 프로그레스 바 숨김
    button_start_check.config(state=tk.NORMAL)
    fast_button_start_check.config(state=tk.NORMAL)
    button_select_file.config(state=tk.NORMAL)
    button_select_folder.config(state=tk.NORMAL)
    button_stop_processing.config(state=tk.DISABLED)

def threaded_similarity_check(args):
    if not selected_png or not selected_folder:
        log_message("PNG or folder not selected.", log_queue=log_queue)
        return    
    def start_processing():
        processed_files.clear()
        listbox_processed_files.delete(0, tk.END)
        button_start_check.config(state=tk.DISABLED)
        fast_button_start_check.config(state=tk.DISABLED)
        button_select_file.config(state=tk.DISABLED)
        button_select_folder.config(state=tk.DISABLED)
        button_stop_processing.config(state=tk.NORMAL)
        # 병렬 처리 함수 실행
        process_folder_parallel(selected_png, selected_folder, on_all_threads_completed, args)
    label_status.config(text=f"calculating")
    tkRoot.update_idletasks()
    processing_thread = threading.Thread(target=start_processing)
    processing_thread.start()    
    
    progress_bar.pack()  # 프로그레스 바 표시
    progress_bar.start()  # 프로그레스 바 시작
    
def fast_threaded_similarity_check(args):
    if not selected_png or not selected_folder:
        log_message("PNG or folder not selected.", log_queue=log_queue)
        return
    def start_processing():
        processed_files.clear()
        listbox_processed_files.delete(0, tk.END)
        button_start_check.config(state=tk.DISABLED)
        fast_button_start_check.config(state=tk.DISABLED)
        button_select_file.config(state=tk.DISABLED)
        button_select_folder.config(state=tk.DISABLED)
        button_stop_processing.config(state=tk.NORMAL)
        # 병렬 처리 함수 실행
        fast_process_folder_parallel(selected_png, selected_folder, on_all_threads_completed, args)
    label_status.config(text=f"calculating")
    tkRoot.update_idletasks()
    processing_thread = threading.Thread(target=start_processing)
    processing_thread.start()    
    
    progress_bar.pack()  # 프로그레스 바 표시
    progress_bar.start()  # 프로그레스 바 시작    

def update_memory_usage():
    # 시스템의 메모리 사용 정보를 가져옵니다.
    memory = psutil.virtual_memory()
    used_memory = memory.used / (1024 ** 3)  # 사용 중인 메모리를 GB 단위로 변환
    total_memory = memory.total / (1024 ** 3)  # 전체 메모리를 GB 단위로 변환
    
    # 메모리 사용량을 라벨에 표시
    memory_label.config(text=f"Memory Usage: {used_memory:.2f} GB / {total_memory:.2f} GB")
    
    # 1초마다 메모리 사용량을 업데이트
    tkRoot.after(1000, update_memory_usage)
    
def stop_processing():
    """병렬 처리 중인 작업을 중지합니다."""
    if 'pool' in globals():
        global pool
        pool.terminate()  # 현재 실행 중인 모든 작업을 중지합니다.
        print("Processing terminated.")
        label_status.config(text="Processing terminated.")    
        button_start_check.config(state=tk.NORMAL)
        fast_button_start_check.config(state=tk.NORMAL)
        button_select_file.config(state=tk.NORMAL)
        button_select_folder.config(state=tk.NORMAL)
        button_stop_processing.config(state=tk.DISABLED)
        progress_bar.stop()  # 프로그레스 바 정지
        progress_bar.pack_forget()  # 프로그레스 바 숨김

def open_in_explorer(event):
    """
    선택된 항목의 파일이 위치한 폴더를 파일 탐색기에서 엽니다.
    Ctrl 키와 함께 리스트 아이템 클릭 시 해당 기능이 활성화됩니다.
    """
    selected_index = listbox_processed_files.curselection()
    if selected_index:
        # 선택된 항목에서 파일 경로 추출
        selected_path = listbox_processed_files.get(selected_index[0]).split("  -  ")[0]

        # 파일이 위치한 폴더 경로를 얻음
        folder_path = os.path.dirname(selected_path)

        try:
            # Windows에서 폴더를 파일 탐색기에서 열기
            os.startfile(folder_path)
        except AttributeError:
            # os.startfile이 없는 macOS와 Linux에서 대안적인 방법
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, folder_path])
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open the folder: {folder_path}\n{e}")

            
def check_for_ctrl_click(event):
    # Ctrl 키가 눌려있는지 확인합니다. (event.state == 4)
    if event.state & 0x0004:
        open_in_explorer(event)
# Tkinter GUI 설정
if __name__ == "__main__":
    multiprocessing.freeze_support()
    merged_dict ={};
    manager = Manager()
    lock = manager.Lock()
    last_path_select = None;
    #last_path_select = manager.Value('S', None)
    processed_files = manager.dict()  # 프로세스 간 공유되는 딕셔너리
    log_queue = manager.Queue()
    #active_threads = 0
    availableProcess = os.cpu_count()          # CPU 전체개수 조회
    # processPool = ProcessPoolExecutor(availableProcess)     # 사용할 CPU 개수 설정, 논리(logical) CPU당 하나의 worker 
    tkRoot = tk.Tk()
    tkRoot.title("Image Similarity Checker")
    
    # paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    # paned_window.pack(fill=tk.BOTH, expand=True)

    # Listbox와 Scrollbar 생성 및 연결
    listbox_processed_files = tk.Listbox(tkRoot, width=40, height=10)
    scrollbar_processed_files = tk.Scrollbar(tkRoot, command=listbox_processed_files.yview)
    listbox_processed_files.config(yscrollcommand=scrollbar_processed_files.set)
    #paned_window.add(listbox_processed_files)
    # Listbox와 Scrollbar 배치
    listbox_processed_files.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_processed_files.pack(side=tk.RIGHT, fill=tk.Y)

    listbox_image_details = tk.Listbox(tkRoot, width=40, height=10)
    listbox_image_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
   
    #paned_window.add(listbox_image_details)

    # 이미지를 표시할 Label 위젯 생성
    image_label = tk.Label(tkRoot)
    image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    # 이미지를 표시할 Label 위젯 생성
    select_image_label = tk.Label(tkRoot)
    select_image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Label for the selected PNG file
    label_selected_file = tk.Label(tkRoot, text="No file selected.")
    label_selected_file.pack()

    # Button to select a PNG file
    button_select_file = tk.Button(tkRoot, text="Select PNG File", command=select_file)
    button_select_file.pack()

    # Label for the selected folder
    label_selected_folder = tk.Label(tkRoot, text="No folder selected.")
    label_selected_folder.pack()

    # Button to select a folder
    button_select_folder = tk.Button(tkRoot, text="Select Folder", command=select_folder)
    button_select_folder.pack()

    args = (log_queue, processed_files, lock)
    log_message("init: " + str(processed_files));
    log_message("init: " + str(log_queue));
    
    def threaded_similarity_check_with_args():
        threaded_similarity_check(args)
    button_start_check = tk.Button(tkRoot, text="Slow Similarity Check", command=threaded_similarity_check_with_args)
    button_start_check.pack()
    
    def fast_threaded_similarity_check_with_args():
        fast_threaded_similarity_check(args)
    fast_button_start_check = tk.Button(tkRoot, text="Fast Similarity Check", command=fast_threaded_similarity_check_with_args)
    fast_button_start_check.pack()
        
    # 중지 버튼 추가
    button_stop_processing = tk.Button(tkRoot, text="Stop Processing", command=stop_processing)
    button_stop_processing.pack()    
    button_stop_processing.config(state=tk.DISABLED)
    
    listbox_image_details.bind('<<ListboxSelect>>', lambda event, arg1=args: on_image_selected(event, arg1))
    listbox_processed_files.bind('<<ListboxSelect>>', lambda event, arg1=args: show_image_details(event, arg1))
    listbox_processed_files.bind('<Button-1>', check_for_ctrl_click)
    # 메모리 사용량을 표시할 라벨 추가
    memory_label = tk.Label(tkRoot, text="Memory Usage: 0 GB / 0 GB")
    memory_label.pack()    

    label_cpu = tk.Label(tkRoot, text="cpu_count: " + str(availableProcess))
    label_cpu.pack() 
    
    # 로딩 바 추가
    progress_bar = ttk.Progressbar(tkRoot, mode='indeterminate')
    progress_bar.pack()
    
    # 작업 상태
    label_status = tk.Label(tkRoot, text=f"ready")
    label_status.pack()
    #tkRoot.bind('<Control-Left>', open_in_explorer)
    #tkRoot.after(1000, lambda: update_log_messages(log_queue))
    update_memory_usage()
    tkRoot.mainloop()