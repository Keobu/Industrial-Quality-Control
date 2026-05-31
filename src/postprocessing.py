import cv2
import numpy as np

class DefectLocalizer:
    @staticmethod
    def highlight_defects(image_path, preprocessed_image):
        
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            return None, False
            
        # Rescaling normalized image back to uint8 [0, 255]
        img_8bit = (preprocessed_image * 255).astype(np.uint8)
        
        # Invert thresholding to segment dark anomalies (scratches)
        _, thresh = cv2.threshold(img_8bit, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Morphology engine: closing filter to bridge disjoint pixel groups
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Extract topological contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defect_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 20:  # Area noise filter
                defect_detected = True
                (x, y, w, h) = cv2.boundingRect(contour)
                # Drawing visual cues on the source frame
                cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(orig_img, "ANOMALY", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        return orig_img, defect_detected