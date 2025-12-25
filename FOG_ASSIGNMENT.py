import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import os

@dataclass
class ImageAnalysis:
    """Stores analysis results for image quality assessment"""
    needs_deblur: bool = False
    needs_sharpening: bool = False
    needs_contrast: bool = False
    needs_lighting: bool = False
    needs_background_blur: bool = True
    blur_score: float = 0.0
    contrast_score: float = 0.0
    brightness_score: float = 0.0
    sharpness_score: float = 0.0

class PortraitEnhancer:
    """Modular portrait enhancement system with intelligent quality detection"""
    
    def __init__(self, debug=True):
        self.debug = debug
        self.applied_modules = []
        
    def analyze_image(self, image: np.ndarray) -> ImageAnalysis:
        """Analyze image to determine which enhancements are needed"""
        analysis = ImageAnalysis()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur Detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        analysis.blur_score = laplacian_var
        analysis.sharpness_score = laplacian_var
        
        # Thresholds: <100 is blurry, <500 needs sharpening
        analysis.needs_deblur = laplacian_var < 100
        analysis.needs_sharpening = 100 <= laplacian_var < 500
        
        # 2. Contrast Analysis
        min_val, max_val = gray.min(), gray.max()
        contrast_range = max_val - min_val
        analysis.contrast_score = contrast_range
        analysis.needs_contrast = contrast_range < 100
        
        # 3. Brightness/Lighting Analysis
        mean_brightness = gray.mean()
        analysis.brightness_score = mean_brightness
        # Underexposed (<80) or overexposed (>200)
        analysis.needs_lighting = mean_brightness < 80 or mean_brightness > 200
        
        # 4. Background blur always recommended for studio look
        analysis.needs_background_blur = True
        
        if self.debug:
            print("\n=== IMAGE ANALYSIS ===")
            print(f"Blur Score (Laplacian Variance): {laplacian_var:.2f}")
            print(f"  → Needs Deblur: {analysis.needs_deblur}")
            print(f"  → Needs Sharpening: {analysis.needs_sharpening}")
            print(f"Contrast Range: {contrast_range:.2f}")
            print(f"  → Needs Contrast Enhancement: {analysis.needs_contrast}")
            print(f"Brightness: {mean_brightness:.2f}")
            print(f"  → Needs Lighting Correction: {analysis.needs_lighting}")
            print("="*30 + "\n")
        
        return analysis
    
    def detect_face_region(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Detect face region for targeted enhancement"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Expand region by 30% for better context
            margin_w, margin_h = int(w * 0.3), int(h * 0.3)
            x = max(0, x - margin_w)
            y = max(0, y - margin_h)
            w = min(image.shape[1] - x, w + 2 * margin_w)
            h = min(image.shape[0] - y, h + 2 * margin_h)
            
            return None, (x, y, w, h)
        
        return None, None
    
    def create_person_mask(self, image: np.ndarray, face_region: Tuple) -> np.ndarray:
        """Create a sophisticated mask to separate person from background"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if face_region is None:
            # No face detected - return empty mask (will blur less aggressively)
            return mask
        
        x, y, w, h = face_region
        
        # Method 1: GrabCut-based segmentation for better person detection
        rect = (max(0, x - w//4), max(0, y - h//2), 
                min(width, w + w//2), min(height, h + h*2))
        
        # Initialize mask for GrabCut
        gc_mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Apply GrabCut algorithm
            cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask (foreground and probable foreground)
            mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
            
        except:
            # Fallback: Use expanded face region and edge detection
            center_x, center_y = x + w//2, y + h//2
            
            # Create elliptical mask covering upper body
            body_width = int(w * 1.8)
            body_height = int(h * 2.5)
            
            cv2.ellipse(mask, (center_x, center_y + h//2), 
                       (body_width, body_height), 0, 0, 360, 1, -1)
        
        # Refine mask with edge-aware filtering
        mask = mask.astype(np.float32)
        
        # Apply guided filter for smooth edges
        radius = 20
        eps = 0.01
        
        # Simple guided filter implementation
        mean_I = cv2.boxFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255, -1, (radius, radius))
        mean_p = cv2.boxFilter(mask, -1, (radius, radius))
        corr_Ip = cv2.boxFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255 * mask, -1, (radius, radius))
        
        var_I = cv2.boxFilter((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255) ** 2, -1, (radius, radius)) - mean_I ** 2
        cov_Ip = corr_Ip - mean_I * mean_p
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        mask = mean_a * (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255) + mean_b
        mask = np.clip(mask, 0, 1)
        
        # Additional feathering for smooth transition
        mask = cv2.GaussianBlur(mask, (31, 31), 10)
        
        return (mask * 255).astype(np.uint8)
    
    def module_deblur(self, image: np.ndarray) -> np.ndarray:
        """Remove motion blur using deconvolution"""
        self.applied_modules.append("Deblur")
        
        # Wiener filter approximation
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        deblurred = cv2.filter2D(image, -1, kernel)
        
        # Bilateral filter to reduce noise while preserving edges
        deblurred = cv2.bilateralFilter(deblurred, 5, 75, 75)
        
        return deblurred
    
    def module_sharpen(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Enhance sharpness with controlled strength"""
        self.applied_modules.append("Sharpen")
        
        # Unsharp mask
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        
        return sharpened
    
    def module_enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adaptive contrast enhancement"""
        self.applied_modules.append("Contrast Enhancement")
        
        # CLAHE on LAB color space to preserve colors
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def module_lighting_correction(self, image: np.ndarray, target_brightness: float = 128) -> np.ndarray:
        """Correct lighting while preserving natural look"""
        self.applied_modules.append("Lighting Correction")
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        current_brightness = l.mean()
        adjustment = target_brightness - current_brightness
        
        # Apply gradual adjustment
        l = np.clip(l + adjustment * 0.5, 0, 255).astype(np.uint8)
        
        corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        
        return corrected
    
    def detect_lighting_imbalance(self, image: np.ndarray, face_region: Tuple) -> bool:
        """Detect if face has uneven lighting (one side shadowed)"""
        if face_region is None:
            return False
        
        x, y, w, h = face_region
        face = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Split face into left and right halves
        mid = w // 2
        left_half = gray_face[:, :mid]
        right_half = gray_face[:, mid:]
        
        left_brightness = left_half.mean()
        right_brightness = right_half.mean()
        
        # Calculate difference
        diff = abs(left_brightness - right_brightness)
        
        # Threshold: if difference > 30, there's significant imbalance
        has_imbalance = diff > 30
        
        if self.debug and has_imbalance:
            print(f"Lighting imbalance detected!")
            print(f"  Left side brightness: {left_brightness:.1f}")
            print(f"  Right side brightness: {right_brightness:.1f}")
            print(f"  Difference: {diff:.1f}")
        
        return has_imbalance
    
    def module_face_lighting_correction(self, image: np.ndarray, face_region: Tuple) -> np.ndarray:
        """Correct uneven lighting on face (half shadow correction)"""
        self.applied_modules.append("Face Lighting Balance")
        
        if face_region is None:
            return image
        
        x, y, w, h = face_region
        face = image[y:y+h, x:x+w].copy()
        
        # Convert to LAB for better lighting manipulation
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create illumination map using guided filter approach
        # Estimate ambient lighting
        illumination = cv2.blur(l, (w//4, h//4))
        illumination = cv2.resize(illumination, (w, h))
        
        # Calculate average illumination
        avg_illumination = np.mean(illumination)
        
        # Create correction map (inverse of illumination variance)
        correction_map = avg_illumination / (illumination.astype(np.float32) + 1)
        correction_map = np.clip(correction_map, 0.5, 2.0)
        
        # Apply correction to L channel
        l_corrected = l.astype(np.float32) * correction_map
        l_corrected = np.clip(l_corrected, 0, 255).astype(np.uint8)
        
        # Smooth transition with CLAHE for natural look
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_enhanced = clahe.apply(l_corrected)
        
        # Blend original and corrected for subtle effect
        l_final = cv2.addWeighted(l, 0.3, l_enhanced, 0.7, 0)
        
        # Merge back
        corrected_lab = cv2.merge([l_final, a, b])
        corrected_face = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
        
        # Place corrected face back with smooth blending at edges
        result = image.copy()
        
        # Create blend mask with feathered edges
        mask = np.zeros((h, w), dtype=np.float32)
        mask[h//8:7*h//8, w//8:7*w//8] = 1.0
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask = np.stack([mask, mask, mask], axis=2)
        
        # Blend face into image
        result[y:y+h, x:x+w] = (corrected_face * mask + face * (1 - mask)).astype(np.uint8)
        
        return result
    
    def module_background_blur(self, image: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Apply professional bokeh effect to background while keeping person sharp"""
        self.applied_modules.append("Background Blur")
        
        if person_mask is None or person_mask.max() == 0:
            # Fallback: gentle blur if no person detected
            blurred = cv2.GaussianBlur(image, (21, 21), 0)
            return cv2.addWeighted(image, 0.6, blurred, 0.4, 0)
        
        # Create high-quality background blur using multiple passes
        # This simulates bokeh effect better than single Gaussian blur
        
        # Pass 1: Strong blur for far background
        blur_strong = cv2.GaussianBlur(image, (0, 0), 15)
        
        # Pass 2: Medium blur for mid-ground
        blur_medium = cv2.GaussianBlur(image, (0, 0), 8)
        
        # Normalize mask to 0-1 range
        mask_norm = person_mask.astype(np.float32) / 255.0
        
        # Create depth-like effect (stronger blur further from person)
        # Invert mask so background = 1, person = 0
        bg_mask = 1 - mask_norm
        
        # Apply distance-based blur intensity
        # Areas closer to person get less blur
        distance_map = cv2.distanceTransform((person_mask < 128).astype(np.uint8), 
                                            cv2.DIST_L2, 5)
        distance_map = cv2.normalize(distance_map, None, 0, 1, cv2.NORM_MINMAX)
        distance_map = cv2.GaussianBlur(distance_map, (51, 51), 0)
        
        # Blend different blur levels based on distance
        # Far areas get strong blur, near areas get medium blur
        bg_mask_3ch = np.stack([bg_mask, bg_mask, bg_mask], axis=2)
        distance_3ch = np.stack([distance_map, distance_map, distance_map], axis=2)
        
        # Blend blur levels
        blurred_bg = (blur_strong * distance_3ch + 
                      blur_medium * (1 - distance_3ch))
        
        # Final composition: sharp person + blurred background
        person_3ch = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
        result = (image * person_3ch + blurred_bg * bg_mask_3ch).astype(np.uint8)
        
        # Apply subtle vignette for studio effect
        rows, cols = image.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/2)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/2)
        
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask_vignette = kernel / kernel.max()
        mask_vignette = np.stack([mask_vignette, mask_vignette, mask_vignette], axis=2)
        
        # Apply vignette only to background
        vignette_strength = 0.3
        result = (result * (1 - bg_mask_3ch * vignette_strength) + 
                 result * mask_vignette * bg_mask_3ch * vignette_strength).astype(np.uint8)
        
        return result
    
    def module_face_enhancement(self, image: np.ndarray, face_region: Tuple) -> np.ndarray:
        """Enhance face clarity without over-processing"""
        self.applied_modules.append("Face Enhancement")
        
        if face_region is None:
            return image
        
        x, y, w, h = face_region
        face = image[y:y+h, x:x+w].copy()
        
        # Subtle skin smoothing
        smoothed = cv2.bilateralFilter(face, 5, 50, 50)
        
        # Detail enhancement
        detail = cv2.detailEnhance(smoothed, sigma_s=10, sigma_r=0.15)
        
        # Blend for natural look
        enhanced_face = cv2.addWeighted(face, 0.3, detail, 0.7, 0)
        
        result = image.copy()
        result[y:y+h, x:x+w] = enhanced_face
        
        return result
    
    def enhance_portrait(self, image_path: str, output_path: str = None) -> np.ndarray:
        """Main enhancement pipeline with intelligent module selection"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}\n")
        
        # Reset applied modules
        self.applied_modules = []
        
        # Step 1: Analyze image
        analysis = self.analyze_image(image)
        
        # Step 2: Detect face
        person_mask, face_region = self.detect_face_region(image)
        
        # Create full person mask for background blur
        if face_region is not None:
            person_mask = self.create_person_mask(image, face_region)
        
        # Step 2.5: Check for lighting imbalance on face
        has_lighting_imbalance = self.detect_lighting_imbalance(image, face_region)
        
        result = image.copy()
        
        # Step 3: Apply modules based on analysis
        if analysis.needs_deblur:
            print("✓ Applying: Deblur")
            result = self.module_deblur(result)
        
        if analysis.needs_lighting:
            print("✓ Applying: Lighting Correction")
            result = self.module_lighting_correction(result)
        
        # Apply face-specific lighting correction if imbalance detected
        if has_lighting_imbalance and face_region is not None:
            print("✓ Applying: Face Lighting Balance (Shadow Correction)")
            result = self.module_face_lighting_correction(result, face_region)
        
        if analysis.needs_contrast:
            print("✓ Applying: Contrast Enhancement")
            result = self.module_enhance_contrast(result)
        
        if analysis.needs_sharpening and not analysis.needs_deblur:
            print("✓ Applying: Sharpening")
            result = self.module_sharpen(result)
        
        if face_region is not None:
            print("✓ Applying: Face Enhancement")
            result = self.module_face_enhancement(result, face_region)
        
        if analysis.needs_background_blur:
            print("✓ Applying: Background Blur (Studio Effect)")
            result = self.module_background_blur(result, person_mask)
        
        # Step 4: Final subtle sharpening for clarity
        result = self.module_sharpen(result, strength=0.3)
        
        # Save output
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_enhanced{ext}"
        
        cv2.imwrite(output_path, result)
        print(f"\n✓ Enhanced image saved: {output_path}")
        print(f"Modules applied: {', '.join(self.applied_modules)}\n")
        
        return result

if __name__ == "__main__":
    enhancer = PortraitEnhancer(debug=True)
    
    input_image = "portrait.jpg"  
    
    if os.path.exists(input_image):
        enhanced = enhancer.enhance_portrait(input_image)
        print("Enhancement complete!")
    else:
        print(f"Sample usage:")
        print(f"  enhancer = PortraitEnhancer(debug=True)")
        print(f"  enhancer.enhance_portrait('your_image.jpg', 'output_enhanced.jpg')")
        print(f"\nPlace your image as 'portrait.jpg' or modify the script.")
    
enhancer = PortraitEnhancer(debug=True)
enhancer.enhance_portrait('camera2.jpg', 'FINAL_OUTPUT_IMAGE2.jpg')      
