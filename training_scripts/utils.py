import cv2 
import numpy as np 
from PIL import Image 
import datetime
import os 
import os.path as osp 
import math 


# def create_image_with_captions(images, captions):
#     # Load images
#     # images = [cv2.imread(path) for path in image_paths]
#     images = [np.array(image).astype(np.uint8) for image in images] 

#     # Check if all images are loaded
#     if any(img is None for img in images):
#         raise ValueError("One or more images could not be loaded. Check the paths.")

#     # Resize images to the same height for proper concatenation
#     height = min(img.shape[0] for img in images)
#     images_resized = [cv2.resize(img, (int(img.shape[1] * (height / img.shape[0])), height)) for img in images]

#     # Create a blank image for captions
#     caption_height = 50  # Height for the caption area
#     caption_images = []

#     for caption in captions:
#         # Create a blank image for the caption
#         caption_img = np.ones((caption_height, images_resized[0].shape[1], 3), dtype=np.uint8) * 255
#         # Put the caption text on the blank image
#         cv2.putText(caption_img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
#         caption_images.append(caption_img)

#     # Concatenate images and captions
#     final_images = [cv2.vconcat([img, caption]) for img, caption in zip(images_resized, caption_images)]
#     final_image = cv2.hconcat(final_images)

#     return Image.fromarray(final_image) 

import cv2
import numpy as np
from PIL import Image

def create_image_with_captions(image_rows, captions):
    # Check if the number of rows matches the number of caption rows
    if len(image_rows) != len(captions):
        raise ValueError("The number of caption rows must match the number of image rows.")

    # Load images and check if they are valid
    images_resized_rows = []
    for row in image_rows:
        images = [np.array(image).astype(np.uint8) for image in row]
        if any(img is None for img in images):
            raise ValueError("One or more images could not be loaded. Check the paths.")
        
        # Resize images to the same height for proper concatenation
        height = min(img.shape[0] for img in images)
        images_resized = [cv2.resize(img, (int(img.shape[1] * (height / img.shape[0])), height)) for img in images]
        images_resized_rows.append(images_resized)

    # Create a blank image for captions
    caption_height = 50  # Height for the caption area
    caption_images = []

    for caption_row in captions:
        caption_images_row = []
        for caption in caption_row:
            # Create a blank image for the caption
            caption_img = np.ones((caption_height, images_resized_rows[0][0].shape[1], 3), dtype=np.uint8) * 255
            # Put the caption text on the blank image
            cv2.putText(caption_img, caption, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA) 
            caption_images_row.append(caption_img)
        caption_images.append(caption_images_row)

    # Concatenate images and captions for each row
    final_rows = []
    for images_resized, caption_images_row in zip(images_resized_rows, caption_images):
        # Concatenate images in the current row
        # for img in images_resized: 
        #     print(f"{img.shape = }")
        final_images = [cv2.vconcat([img, caption]) for img, caption in zip(images_resized, caption_images_row)]
        final_row = cv2.hconcat(final_images)
        final_rows.append(final_row)

    # Concatenate all rows vertically
    final_image = cv2.vconcat(final_rows)

    return Image.fromarray(final_image)


def get_common_seed(): 

    # Get the current date and time
    current_date = datetime.datetime.now()

    # Convert to integer format YYYYMMDDHHMMSS
    date_as_integer = int(current_date.strftime("%d"))

    print("Date and Time in Integer Format:", date_as_integer)

    return date_as_integer 


def create_gif(images, save_path, duration=1):
    """
    Convert a sequence of NumPy array images to a GIF.
    
    Args:
        images (list): A list of NumPy array images.
        fps (int): The frames per second of the GIF (default is 1).
        loop (int): The number of times the animation should loop (0 means loop indefinitely) (default is 0).
    """
    frames = []
    for img in images:
        # Convert NumPy array to PIL Image
        # img_pil = Image.fromarray(img.astype(np.uint8))
        img_pil = img 
        # Append to frames list
        frames.append(img_pil)
    
    # Save frames to a BytesIO object
    # bytes_io = BytesIO()
    # frames[0].save(bytes_io, save_all=True, append_images=frames[1:], duration=1000/fps, loop=loop, 
                #    disposal=2, optimize=True, subrectangles=True)
    dirname = osp.dirname(save_path) 
    if dirname != "": 
        os.makedirs(osp.dirname(save_path), exist_ok=True) 
    frames[0].save(save_path, save_all=True, append_images=frames[1:], loop=0, duration=int(duration * 1000))
    
    # gif_bytes = bytes_io.getvalue()
    # with open("temp.gif", "wb") as f:
    #     f.write(gif_bytes)
    # return gif_bytes 
    return 


def collect_generated_images(vis_dir, prompt, save_path):  
    print(f"received {save_path = } and {vis_dir = }") 
    # stores all the videos for this particular prompt on wandb  
    template_prompt_videos = {} 

    # collecting results of type 1 inference 
    # prompts_dataset = prompts_dataset1 
    # template_prompt_videos["type1"] = {} 
    for subjects_string in sorted(os.listdir(vis_dir)): 

        # save_path_global = osp.join(vis_dir) 

        # subject_prompt = prompt.replace("SUBJECT", subject)   
        # prompt_ = "_".join(subject_prompt.split()) 
        # prompt_path = osp.join(save_path_global, prompt_) 
        imgs_path = osp.join(vis_dir, subjects_string) 
        img_names = os.listdir(imgs_path)  
        img_names = [img_name for img_name in img_names if img_name.find(f"jpg") != -1] 
        img_names = sorted(img_names) 

        # assert "bnha" in subject 

        # template_prompt_videos["type1"][keyname] = [] 
        template_prompt_videos[subjects_string] = [] 
        # assert len(img_names) == prompts_dataset.num_samples, f"{len(img_names) = }, {prompts_dataset.num_samples = }" 
        for img_name in img_names: 
            # print(f"for {subject} i am using {prompt_path = } and {img_name = }") 
            img_path = osp.join(imgs_path, img_name) 
            got_image = False 
            # while not got_image: 
            #     try: 
            #         img = Image.open(img_path) 
            #         got_image = True 
            #     except Exception as e: 
            #         print(f"could not read the image, will try again, don't worry, just read and chill!") 
            #         got_image = False 
            #     if got_image: 
            #         break 
            img = Image.open(img_path) 
            template_prompt_videos[subjects_string].append(img) 


        # concatenate all the images for this template prompt 
    all_concat_imgs = [] 
        # save_path_global = osp.join(args.vis_dir, f"__{args.run_name}", f"outputs_{step_number}")  
        # for idx in range(prompts_dataset.num_samples): 
        #     images = [] 
        #     for subject in sorted(prompts_dataset.subjects): 
        #         images.append(template_prompt_videos[subject][idx]) 
        #     concat_img = create_image_with_captions(images, sorted(prompts_dataset.subjects))  
        #     all_concat_imgs.append(concat_img) 
    num_samples = len(list(template_prompt_videos.values())[0]) 
    for idx in range(num_samples): 
        images = [] 
        captions = [] 
        # for typename in list(template_prompt_videos.keys()): 
        images_row = [] 
        captions_row = [] 
        for subjects_string in list(template_prompt_videos.keys()): 
            images_row.append(template_prompt_videos[subjects_string][idx]) 
            captions_row.append(subjects_string)  
        images.append(images_row)  
        captions.append(captions_row) 

        concat_img = create_image_with_captions(images, captions)  
        concat_img = create_image_with_captions([[concat_img]], [[f"{prompt}, azimuth = {(idx * 2 * math.pi) / num_samples}"]]) 
        all_concat_imgs.append(concat_img) 


    # template_prompt_ = "_".join(template_prompt.split()) 
    # video_path = osp.join(save_path_global, template_prompt_ + ".gif")  
    print(f"sending {save_path = }")
    create_gif(all_concat_imgs, save_path, 1) 