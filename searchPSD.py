import tkinter as tk
import os
import cv2
import numpy as np
import sys
import threading
from tkinter import filedialog
from psd_tools import PSDImage
from PIL import Image, ImageTk
from tkinter import ttk
from threading import Thread

last_path_select = None;
processed_files = {}  # 처리된 파일들의 경로를 저장할 리스트
active_threads = 0

def log_message(*args):
    """GUI의 Label에 로그 메시지를 표시하는 함수. 메시지가 길 경우 두 줄로 나누어 표시."""
    message = " ".join(str(arg) for arg in args)
    # 메시지의 길이가 50자를 초과하는 경우, 두 줄로 나누어 표시
    if len(message) > 50:
        # 가능한 중간 지점을 찾습니다. 예시에서는 간단히 메시지를 반으로 나눕니다.
        # 더 정교한 접근 방식을 원한다면, 단어 경계에서 나누는 것을 고려할 수 있습니다.
        split_point = len(message) // 2
        # 중간 지점에서 가장 가까운 공백을 찾아 그 지점에서 줄을 나눕니다.
        nearest_space = message.rfind(' ', 0, split_point)
        if nearest_space != -1:
            # 가장 가까운 공백 위치를 기준으로 줄바꿈
            message = message[:nearest_space] + '\n' + message[nearest_space+1:]
        else:
            # 공백이 없는 경우, 강제로 지정된 위치에서 줄을 나눕니다.
            message = message[:split_point] + '\n' + message[split_point:]
    label_status.config(text=message)
    tkRoot.update()  # GUI를 즉시 업데이트


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
        
def on_image_selected(event):
    """ 사용자가 이미지를 선택했을 때 실행되는 함수 """
    selected_index = listbox_image_details.curselection()
    log_message(f"selected_index: {selected_index}")
    
    if not selected_index:
        return

    selected_text = listbox_image_details.get(selected_index)
    selected_image_name = selected_text.split(" - ")[0]
    log_message(f"selected_image_name: {selected_image_name}")
    
    # listbox_processed_files에서 현재 선택된 항목의 인덱스를 확인
    # processed_files_selected_index = listbox_processed_files.curselection()
    # log_message(f"processed_files_selected_index: {processed_files_selected_index}")
    # if not processed_files_selected_index:
    #     return

    # selected_path_text = listbox_processed_files.get(processed_files_selected_index)
    # selected_path = selected_path_text.split(" - ")[0]
    log_message(f"selected_path: {last_path_select}")
    # 이미지 파일의 경로를 가져와서 표시
    if last_path_select in processed_files and selected_image_name in processed_files[last_path_select]:
        image = processed_files[last_path_select][selected_image_name]["image"]
        show_selected_image(image)


def show_image_details(event):
    """ 선택된 경로에 있는 이미지들의 상세 정보를 표시하는 함수 """
    selected_index = listbox_processed_files.curselection()
    if not selected_index:
        return

    selected_text = listbox_processed_files.get(selected_index[0])
    selected_path = selected_text.split(" - ")[0]
    global last_path_select
    last_path_select = selected_path

    listbox_image_details.delete(0, tk.END)

    if selected_path in processed_files:
        # 유사도에 따라 정렬
        sorted_images = sorted(processed_files[selected_path].items(), key=lambda x: x[1]['similarity'], reverse=True)
        for image_name, image_info in sorted_images:
            listbox_image_details.insert(tk.END, f"{image_name} - Similarity: {image_info['similarity']:.2f}")



def start_similarity_check(callback):
    """ 유사도 검사를 시작하는 함수 """
    global processed_files
    processed_files = {}  # 리스트 초기화
    #listbox_processed_files.delete(0, tk.END)
    log_message("start_similarity_check Start")
    if selected_png and selected_folder:
        process_folder_parallel(selected_png, selected_folder)
        # 유사도에 따라 리스트를 정렬
        #processed_files.sort(key=lambda x: x[1], reverse=True)
    else:
        log_message("PNG 파일 또는 폴더가 선택되지 않았습니다.")
    
    log_message("start_similarity_check End")    
    show_processed_files()
    callback()  # 작업 완료 후 콜백 함수 호출
    
def on_similarity_check_completed():
    """유사도 검사 완료 후 실행될 콜백 함수"""
    tkRoot.after(0, progress_bar.stop)  # 메인 스레드에서 프로그레스 바 정지
    tkRoot.after(0, progress_bar.pack_forget)  # 프로그레스 바 숨김

def threaded_similarity_check():
    """유사도 검사를 별도의 스레드에서 실행하는 함수"""
    progress_bar.pack()  # 프로그레스 바 표시
    progress_bar.start()  # 프로그레스 바 시작
    # 백그라운드 스레드에서 start_similarity_check 함수 실행 및 콜백 전달
    threading.Thread(target=lambda: start_similarity_check(on_similarity_check_completed)).start()

def show_processed_files():
    global processed_files
    if processed_files:
        log_message(f"processed_files Size:", len(processed_files))
    else:
        return;
    
    highest_similarity_per_path = []
    for path, images in processed_files.items():
        highest_similarity = 0
        selected_image = None
        for image_name, image_info in images.items():
            x = image_info["similarity"]
            log_message(f"{path} ------ {image_name} ---- Similarity: {x}")
            if image_info["similarity"] > highest_similarity:
                highest_similarity = image_info["similarity"]
                selected_image = (path, image_name, highest_similarity)
        if selected_image:
            highest_similarity_per_path.append(selected_image)
        
    sorted_by_similarity = sorted(highest_similarity_per_path, key=lambda x: x[2], reverse=True)

    for path, image_name, similarity in sorted_by_similarity:
        log_message(f"{path} Similarity: {similarity}")
        listbox_processed_files.insert(tk.END, f"{path} - Similarity: {similarity:.2f}")
        
        
        # - ImageName : {image_name}
    
    # processed_files = dict(sorted(processed_files.items(), key=lambda item: item[1], reverse=True))
    # listbox_processed_files.delete(0, tk.END)  # 기존 리스트 항목을 모두 삭제합니다.
    # for file, similarity in processed_files.items():
    #     log_message(f"{file} Similarity: {similarity}")
    #     listbox_processed_files.insert(tk.END, f"{file} - Similarity: {similarity:.2f}")
        
def safe_path(path):
    """ 파일 경로에서 비-ASCII 문자를 그대로 유지하고, 파일 존재 여부를 확인하는 함수 """
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(path):
            log_message(f"File does not exist: {path}")
            return None
        return path
    except Exception as e:
        log_message(f"Error processing path: {path}, Error: {e}")
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


def process_psd_file(psd_file_path, compare_with):
    """ PSD 파일을 처리하는 함수 """
    safe_psd_file_path = safe_path(psd_file_path)
    if not safe_psd_file_path:
        log_message(f"Error processing PSD file path: {psd_file_path}")
        return

    mergered_img_list = []
    psd = PSDImage.open(safe_psd_file_path)
    for layer in psd:
        if not layer.visible:
            continue
        merged_image = merge_group_layers(psd, layer, safe_psd_file_path, compare_with)
        mergered_img_list.append(merged_image)
        if merged_image:
            log_message("################# Merge #################")
            merged_image_name = f"{layer.name}.png"
            #merged_image.save(merged_image_name)
            compare_images_in_memory(compare_with, merged_image, safe_psd_file_path, merged_image_name)
    
    max_width = psd.width
    max_height = psd.height
    empty_image = Image.new('RGBA', (max_width, max_height))        
    for merged in reversed(mergered_img_list):
        empty_image = Image.alpha_composite(merged, empty_image)
        
    log_message("################# Merged Merge #################")
    merged_image_name_Merged = f"{layer.name}_Merged.png"
    #empty_image.save(merged_image_name_Last)
    compare_images_in_memory(compare_with, empty_image, safe_psd_file_path, merged_image_name_Merged)     
    merged_image_name_Merged_Template = f"{layer.name}_Search_In_Merged_Template_Image.png"  
    position, value = find_template_in_image(compare_with, empty_image, safe_psd_file_path, merged_image_name_Merged_Template);
    

# def select_file():
#     """ Let the user select a PNG file. """
#     file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
#     file_path = os.path.normpath(file_path)
#     if file_path:
#         global selected_png
#         selected_png = file_path
#         label_selected_file.config(text=f"Selected PNG File: {file_path}")
#     else:
#         label_selected_file.config(text="No file selected.")

def select_file():
    """ Let the user select a PNG file. """
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
    file_path = os.path.normpath(file_path)
    if file_path:
        global selected_png
        selected_png = file_path
        label_selected_file.config(text=f"Selected PNG File: {file_path}")
        
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
        label_selected_folder.config(text=f"Selected Folder: {folder_path}")
        #process_folder(selected_png, folder_path)
    else:
        label_selected_folder.config(text="No folder selected.")

# def process_folder(png_file, folder_path):
#     """ Process all images in the selected folder. """
#     for filename in os.listdir(folder_path):
#         if filename.endswith((".png", ".jpg", ".jpeg")):
#             image_path = os.path.join(folder_path, filename)
#             compare_images(png_file, image_path)

def process_folder(png_file, folder_path):
    """ 지정된 폴더 및 하위 폴더에 있는 모든 파일을 처리합니다. """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.endswith(".psd"):
                process_psd_file(file_path, png_file)
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                compare_images_pil(png_file, file_path, filename)



def imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    try:
        # 파일을 바이트로 읽고 numpy 배열로 변환
        data = np.fromfile(filename, dtype=np.uint8)
        # cv2.imdecode 함수를 사용하여 이미지 로드
        return cv2.imdecode(data, flags)
    except Exception as e:
        log_message(e)
        return None

def update_dict(my_dict, path, imageName, image, similarity):
    if path not in my_dict:
        my_dict[path] = {}

    if imageName not in my_dict[path]:
        my_dict[path][imageName] = {"similarity": similarity, "image": image}
    
def update_dict_if_higher(my_dict, key, new_value):
    if key not in my_dict or my_dict[key] < new_value:
        my_dict[key] = new_value    

def find_template_in_image(image_path, mergedImage, PsdPath, NamePath):
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
    update_dict(processed_files, PsdPath, NamePath, mergedImage, similarity);
    
    return top_left, max_val

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

def compare_images_in_memory(image1_path, image2, PsdPath, NamePath):
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
    #processed_files.append((psd_file_path, similarity))
    #listbox_processed_files.insert(tk.END, f"{psd_file_path} - Similarity: {similarity:.2f}")
    #update_dict_if_higher(processed_files, PsdPath, similarity);
    update_dict(processed_files, PsdPath, NamePath, none_convert, similarity);
    #merged_image_name = f"{safe_psd_file_path}_{layer.name}.png"
    merged_resize_image1_name = f"{PsdPath}_resize_1.png"
    merged_resize_image2_name = f"{PsdPath}_resize_2.png"
    
    # image1_resized.save(merged_resize_image1_name);
    # image2_resized.save(merged_resize_image2_name);

def compare_images_pil(image1_path, image2_path, ImageName):
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
    #processed_files.append((image2_path, similarity))
    #listbox_processed_files.insert(tk.END, f"{image2_path} - Similarity: {similarity:.2f}")
    #update_dict_if_higher(processed_files, image2_path, similarity);
    update_dict(processed_files, image2_path, ImageName, image2_origin, similarity);
    
def update_thread_count():
    """현재 활성화된 스레드 개수를 업데이트하는 함수"""
    label_threads_count.config(text=f"Active threads: {active_threads}") 
    
def process_folder_parallel(png_file, folder_path, callback):
    """지정된 폴더 및 하위 폴더에 있는 모든 파일을 병렬로 처리합니다."""
    threads = []
    def process_file(file_path):
        if file_path.endswith(".psd"):
            process_psd_file(file_path, png_file)
        elif file_path.endswith((".png", ".jpg", ".jpeg")):
            compare_images_pil(png_file, file_path, os.path.basename(file_path))
        # 필요한 경우 여기에 스레드 완료 후 처리 로직을 추가할 수 있습니다.
        # 스레드 완료 후
        global active_threads
        active_threads -= 1
        update_thread_count()

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            thread = threading.Thread(target=process_file, args=(file_path,))
            threads.append(thread)
            thread.start()
            global active_threads
            active_threads += 1
            update_thread_count()
            label_status.config(text=f"Processing: {file_path}")

    for thread in threads:
        thread.join()

    # 스레드가 모두 완료된 후 실행할 콜백 함수, 예를 들어 GUI 업데이트
    tkRoot.after(0, callback)
    if active_threads == 0:
        label_status.config(text="Complete")  # 모든 작업 완료

def on_all_threads_completed():
    """모든 스레드가 완료된 후 실행할 함수"""
    log_message("show_processed_files.")
    show_processed_files()
    global active_threads
    active_threads = 0  # 스레드 개수 리셋
    update_thread_count()
    label_status.config(text="Completed")
    log_message("Completed.")
    # 여기에 필요한 모든 완료 후 처리 로직을 추가합니다.
    progress_bar.stop()  # 프로그레스 바 정지
    progress_bar.pack_forget()  # 프로그레스 바 숨김

def threaded_similarity_check():
    """유사도 검사를 별도의 스레드에서 실행하는 함수 (병렬 처리)"""
    if not selected_png or not selected_folder:
        log_message("PNG 파일 또는 폴더가 선택되지 않았습니다.")
        return

    progress_bar.pack()  # 프로그레스 바 표시
    progress_bar.start()  # 프로그레스 바 시작
    # 병렬 처리 함수 실행 및 완료 후 콜백 전달
    threading.Thread(target=lambda: process_folder_parallel(selected_png, selected_folder, on_all_threads_completed)).start()
   
    
# def compare_images(image1_path, image2_path):
#     """ Compare two images and log_message their similarity. """
#     # 이미지를 읽습니다.
#     image1 = check_and_load_image(image1_path)
#     image2 = check_and_load_image(image2_path)

#     # 이미지가 유효한지 확인합니다.
#     if image1 is None or image2 is None:
#         log_message(f"Error reading one of the images: {image1_path} or {image2_path}")
#         log_message(f"Error reading one of the images: {image1} or {image2}")
#         return  # 이미지 읽기에 실패하면 함수를 종료합니다.

#     # Resize images to the same size
#     image1 = cv2.resize(image1, (100, 100))
#     image2 = cv2.resize(image2, (100, 100))

#     # Calculate the similarity
#     similarity = ssim(image1, image2)
#     log_message(f"Similarity between {image1_path} and {image2_path}: {similarity}")
#     # processed_files 리스트에 유사도와 경로를 추가합니다.
#     #((image2_path, similarity))
#     #listbox_processed_files.insert(tk.END, f"{image2_path} - Similarity: {similarity:.2f}")
#     #update_dict_if_higher(processed_files, image2_path, similarity);
#     update_dict(processed_files, image2_path, image2_path, image2, similarity);

# Tkinter GUI 설정
if __name__ == "__main__":
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
    listbox_processed_files.bind('<<ListboxSelect>>', show_image_details)
    #paned_window.add(listbox_image_details)

    # 이미지를 표시할 Label 위젯 생성
    image_label = tk.Label(tkRoot)
    image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    # 이미지를 표시할 Label 위젯 생성
    select_image_label = tk.Label(tkRoot)
    select_image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    listbox_image_details.bind('<<ListboxSelect>>', on_image_selected)

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

    button_start_check = tk.Button(tkRoot, text="Start Similarity Check", command=start_similarity_check)
    button_start_check.config(command=threaded_similarity_check)
    button_start_check.pack()
    
    # command=lambda: on_start_button_click(root, selected_png, selected_folder))

    # 로딩 바 추가
    progress_bar = ttk.Progressbar(tkRoot, mode='indeterminate')
    progress_bar.pack()

    # 작업 상태와 스레드 개수를 표시할 Label 위젯
    label_status = tk.Label(tkRoot, text="Ready")
    label_status.pack()

    # 현재 실행 중인 스레드 개수를 표시할 Label 위젯
    label_threads_count = tk.Label(tkRoot, text="Active threads: 0")
    label_threads_count.pack()

    log_message("Default encoding:", sys.getdefaultencoding())

    tkRoot.mainloop()