"""
📌 JND-Based Watermarking System for Video Protection
-----------------------------------------------------
🔹 Author: Andrei Sorin Ștefan & ChatGPT
🔹 Description:
    This module applies **Just Noticeable Difference (JND) watermarking** to videos,
    ensuring that content remains protected even if screen-captured or cropped.
    
    Features:
    - **Pseudo-random watermark embedding** using a star-shaped mask.
    - **SIFT-based feature extraction** for high-density watermarking.
    - **JND-based watermark modulation** for adaptive intensity changes.
    - **Bit sequence decoding** to verify embedded IDs.
    
🔹 Dependencies:
    - OpenCV, NumPy, Secrets, SIFT
    
🔹 Usage:
    - Import `JndWatermark` and use `.embed_pseudo_random_id_color()` to apply a watermark.
"""


import cv2
import numpy as np
from .onnx_inference import inference
import secrets

def draw_star_mask(size):
    """Creează o mască în formă de stea pentru watermark"""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = size // 3

    # Desenăm steaua
    points = np.array([
        [center[0], center[1] - radius],
        [center[0] + radius, center[1] + radius],
        [center[0] - radius, center[1] + radius]
    ], np.int32)

    cv2.fillPoly(mask, [points], (255, 255, 255))
    return mask

class JndWatermark:
    def __init__(self, region_size=35, strength=20):  # Creștem efectul watermark-ului
        self.region_size = region_size
        self.strength = strength

    def embed_pseudo_random_id_color(self, image, pseudo_id):
        """ Aplică watermark-ul pe întreaga imagine folosind steluțe """
        result = image.copy()
        height, width = image.shape[:2]
    
        bit_sequence = [int(bit) for bit in f"{pseudo_id:032b}"]
        bit_idx = 0
    
        # Generăm o mască în formă de stea
        star_mask = draw_star_mask(self.region_size)
    
        for y in range(0, height, self.region_size):
            for x in range(0, width, self.region_size):
                if bit_idx >= 32:
                    bit_idx = 0  # Resetăm indexul dacă depășim 32 de biți
                
                bit = bit_sequence[bit_idx]
    
                for c in range(3):  # Procesăm fiecare canal (R, G, B)
                    region_block = result[y:y+self.region_size, x:x+self.region_size, c]
                    jnd_map = self.calculate_jnd(region_block)
    
                    # Aplicăm watermark-ul în formă de stea
                    mod_value = jnd_map.mean() * (self.strength if bit else -self.strength)
    
                    # Aplicăm masca de stea doar în acea zonă
                    star_applied = cv2.bitwise_and(region_block, region_block, mask=star_mask)
                    star_applied = np.clip(star_applied + mod_value, 0, 255).astype(np.uint8)
    
                    result[y:y+self.region_size, x:x+self.region_size, c] = star_applied
    
                bit_idx += 1
    
        return result
    

    def calculate_jnd(self, image_channel):
        """ Calculează sensibilitatea JND pentru luminanță și textură """
        luminance_threshold = 5
        texture_threshold = 10
        jnd_map = np.zeros_like(image_channel, dtype=np.float32)

        # Luminance sensitivity
        jnd_map += np.clip(image_channel / 255.0 * luminance_threshold, 0, luminance_threshold)

        # Texture sensitivity
        epsilon = 1e-10
        grad_x = cv2.Sobel(image_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        jnd_map += np.clip(grad_magnitude / (grad_magnitude.max() + epsilon) * texture_threshold, 0, texture_threshold)

        return jnd_map



    # Extract Feature Points using SIFT
    def extract_sift_features(self, image_gray):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image_gray, None)
        return keypoints, descriptors

    # Construct Embedding Regions Based on SIFT Density
    def get_dense_feature_regions(self, image, keypoints, is_sort=True):
        h, w = image.shape[:2]
        density_map = np.zeros((h // self.region_size, w // self.region_size))
        
        # Map keypoints to regions
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            density_map[y // self.region_size-1, x // self.region_size-1] += 1
        
        # Identify regions with high density
        dense_regions = np.argwhere(density_map > np.mean(density_map))

        dense_regions_sorted = sorted(
            dense_regions, 
            key=lambda region: density_map[region[0], region[1]], 
            reverse=True
        )
        
        if is_sort==False:
            np.random.shuffle(dense_regions_sorted)
        return dense_regions_sorted[:32]


    # JND Model for Embedding Guidance
    def calculate_jnd(self, image_channel):
        luminance_threshold = 5
        texture_threshold = 10
        jnd_map = np.zeros_like(image_channel, dtype=np.float32)
        
        # Luminance sensitivity
        jnd_map += np.clip(image_channel / 255.0 * luminance_threshold, 0, luminance_threshold)
        
        # Texture sensitivity
        epsilon = 1e-10
        grad_x = cv2.Sobel(image_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_channel, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        jnd_map += np.clip(grad_magnitude / (grad_magnitude.max()+epsilon) * texture_threshold, 0, texture_threshold)
        
        return jnd_map

    def init_regions(self, image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # SIFT feature extraction
        keypoints, _ = self.extract_sift_features(image_gray)
        # Identify dense regions
        dense_regions = self.get_dense_feature_regions(image_gray, keypoints)
        return dense_regions

    def embed_pseudo_random_id_color(self, image, pseudo_id, dense_regions):
        watermarked_image = image.copy()
        bit_sequence = [int(bit) for bit in f"{pseudo_id:032b}"]  # Convert to 32-bit binary
        bit_idx = 0
        
        print(f"📌 Dimensiunea listei `dense_regions`: {len(dense_regions)}")
        print(f"🔍 Valoarea lui `bit_idx`: {bit_idx}")
        if bit_idx >= len(dense_regions):
            raise IndexError(f"❌ `bit_idx` ({bit_idx}) depășește dimensiunea `dense_regions` ({len(dense_regions)})!")
        for bit_idx in range(32):

            y1, x1 = dense_regions[bit_idx] * self.region_size
            for c in range(3):  # Process each channel (R, G, B)
                half_region_size=int(self.region_size/2)
                region_block_1 = watermarked_image[y1:y1 + half_region_size, x1:x1 + self.region_size, c]
                
                jnd_map_1 = self.calculate_jnd(region_block_1)

                # Modulate intensity based on the bit
                bit = bit_sequence[bit_idx]
                # mod_value_1 =  (jnd_map_1.mean() *strength if bit else -255)
                mod_value_1 =  jnd_map_1.mean() *(self.strength if bit else -1*self.strength)
                region_block_1=region_block_1.astype(np.int32)
                watermarked_image[y1:y1 + half_region_size, x1:x1 + self.region_size, c] = np.clip(
                    region_block_1 + mod_value_1, 0, 255
                ).astype(np.uint8)

        return watermarked_image

    
    def generate_embeded_result(self, image, pseudo_id):
        
        # Embed pseudo-random ID to head
        hair_image, hair_mask, face_pixels = inference(image)
        dense_regions = self.init_regions(hair_image)
        watermarked_image = self.embed_pseudo_random_id_color(hair_image, pseudo_id, dense_regions)

        # Embed pseudo random ID to background image
        self.region_size=30
        inverted_mask = cv2.bitwise_not(hair_mask)
        original_image = cv2.bitwise_and(image, image, mask=inverted_mask)
        odense_regions = self.init_regions(original_image)
        owatermarked_image = self.embed_pseudo_random_id_color(original_image, pseudo_id, odense_regions)

        # Merge background and head
        result = cv2.add(owatermarked_image, watermarked_image)
        # Save and display the watermarked image
        # cv2.imwrite(f"watermarked_image_{pseudo_id}.jpg", result)
        return result

    def decode_pseudo_random_id(self, watermarked_image, dense_regions, region_size):
        bit_sequence = []

        for bit_idx in range(32):
            half_region_size=int(region_size/2)
            # Get the top-left coordinates of the region
            y1, x1 = dense_regions[bit_idx] * region_size

            # Initialize bit for this region
            channel_bits = []

            for c in range(3):  # Process each channel (R, G, B)
                region_block_1 = watermarked_image[y1:y1 + half_region_size, x1:x1 + region_size, c]

                region_block_2 = watermarked_image[y1 + half_region_size: y1 + region_size, x1:x1 + region_size, c]
                # Decode the bit from the average intensity of the region
                intensity1 = region_block_1.mean()
                intensity2 = region_block_2.mean()

                # Infer the bit (threshold around 50 for positive and negative modulation)
                bit = 1 if intensity1 > intensity2  else 0
                channel_bits.append(bit)

            # Use majority voting to determine the bit for this region
            bit_sequence.append(1 if sum(channel_bits) > 1 else 0)

        # Convert bit sequence to integer
        decoded_id = int("".join(map(str, bit_sequence)), 2)
        return decoded_id
    
    def decode_max_match_id(self, image, aligned_image, id_list):
        print("Decoding jnd watermark for screen photography...")
        hair_image, hair_mask = inference(image)
        dense_regions=self.init_regions(hair_image)
        # Decode the pseudo-random ID
        decoded_id = self.decode_pseudo_random_id(aligned_image, dense_regions, self.region_size)

        matched_percentage, matched_id=find_match_id(decoded_id, id_list)
        print(f"Decoded User ID : {decoded_id}")
        if matched_percentage>80:
            return matched_id, matched_percentage
        else:
            return 0, 0

def compare_bit_sequence(pseudo_id_1, pseudo_id_2):        
    # Convert to 32-bit binary sequences
    bit_sequence_1 = [int(bit) for bit in f"{pseudo_id_1:032b}"]
    bit_sequence_2 = [int(bit) for bit in f"{pseudo_id_2:032b}"]
    # Compare bit sequences element-wise
    matching_bit_cnt = sum(bit1 == bit2 for bit1, bit2 in zip(bit_sequence_1, bit_sequence_2))
    return matching_bit_cnt, bit_sequence_2

def find_match_id(decoded_id, id_list):
    max_match_cnts=0
    max_match_id_sequence=0
    for user_id in id_list:
        match_cnts, match_id_sequence=compare_bit_sequence(decoded_id, user_id)
        if match_cnts>max_match_cnts:
            max_match_cnts=match_cnts
            max_match_id_sequence=match_id_sequence
    max_match_id= int("".join(map(str, max_match_id_sequence)), 2)
    # Calculate percentage
    total_bits = len(match_id_sequence)
    similarity_percentage = (max_match_cnts / total_bits) * 100
    return similarity_percentage, max_match_id



