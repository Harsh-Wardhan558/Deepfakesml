import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from predict import DeepFakePredictor
import os

class DeepFakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFake Detector - Minor Project")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')
        
        # Initialize model
        self.predictor = DeepFakePredictor()
        
        # Create test images if they don't exist
        if not os.path.exists('gui_real.jpg') or not os.path.exists('gui_fake.jpg'):
            self.create_test_images()
        
        self.setup_ui()
    
    def create_test_images(self):
        """Create test images for the GUI"""
        # Simple real face
        real_face = np.full((300, 300, 3), 200, dtype=np.uint8)
        cv2.circle(real_face, (150, 150), 80, (255, 220, 180), -1)
        cv2.circle(real_face, (120, 130), 12, (0, 0, 0), -1)
        cv2.circle(real_face, (180, 130), 12, (0, 0, 0), -1)
        cv2.ellipse(real_face, (150, 180), (25, 12), 0, 0, 180, (0, 0, 0), 4)
        cv2.imwrite('gui_real.jpg', real_face)
        
        # Simple fake face
        fake_face = np.full((300, 300, 3), 150, dtype=np.uint8)
        cv2.rectangle(fake_face, (70, 70), (230, 230), (220, 200, 170), -1)
        cv2.rectangle(fake_face, (110, 120), (130, 140), (0, 0, 0), -1)
        cv2.rectangle(fake_face, (170, 120), (190, 140), (0, 0, 0), -1)
        cv2.line(fake_face, (120, 190), (180, 190), (0, 0, 0), 5)
        noise = np.random.randint(0, 40, (300, 300, 3), dtype=np.uint8)
        fake_face = cv2.add(fake_face, noise)
        cv2.imwrite('gui_fake.jpg', fake_face)
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="🧠 DeepFake Detection System", 
                              font=('Arial', 20, 'bold'), 
                              fg='white', bg='#2c3e50')
        title_label.pack(pady=20)
        
        # Description
        desc_label = tk.Label(self.root, 
                             text="Upload an image or use test samples to detect deepfakes",
                             font=('Arial', 12), 
                             fg='#ecf0f1', bg='#2c3e50')
        desc_label.pack(pady=10)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(button_frame, text="📁 Upload Image", 
                              command=self.upload_image,
                              font=('Arial', 12), 
                              bg='#3498db', fg='white',
                              width=15, height=2)
        upload_btn.grid(row=0, column=0, padx=10)
        
        # Test Real button
        real_btn = tk.Button(button_frame, text="✅ Test Real", 
                            command=lambda: self.analyze_image('gui_real.jpg'),
                            font=('Arial', 12), 
                            bg='#2ecc71', fg='white',
                            width=15, height=2)
        real_btn.grid(row=0, column=1, padx=10)
        
        # Test Fake button
        fake_btn = tk.Button(button_frame, text="❌ Test Fake", 
                            command=lambda: self.analyze_image('gui_fake.jpg'),
                            font=('Arial', 12), 
                            bg='#e74c3c', fg='white',
                            width=15, height=2)
        fake_btn.grid(row=0, column=2, padx=10)
        
        # Image display frame
        image_frame = tk.Frame(self.root, bg='#2c3e50')
        image_frame.pack(pady=20)
        
        # Image display
        self.image_label = tk.Label(image_frame, bg='#34495e', width=400, height=400)
        self.image_label.pack()
        
        # Result display
        self.result_label = tk.Label(self.root, text="", 
                                    font=('Arial', 18, 'bold'), 
                                    bg='#2c3e50', fg='white')
        self.result_label.pack(pady=10)
        
        # Confidence display
        self.confidence_label = tk.Label(self.root, text="", 
                                        font=('Arial', 14), 
                                        bg='#2c3e50', fg='#bdc3c7')
        self.confidence_label.pack(pady=5)
        
        # Instructions
        instructions = tk.Label(self.root, 
                               text="💡 Tip: Use the 'Test Real' and 'Test Fake' buttons for quick demonstration",
                               font=('Arial', 10), 
                               fg='#95a5a6', bg='#2c3e50')
        instructions.pack(pady=10)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def analyze_image(self, image_path):
        try:
            # Display image first
            self.display_image(image_path)
            
            # Analyze the image
            result = self.predictor.predict_image(image_path)
            
            # Display results
            if 'error' in result:
                self.show_error(result['error'])
            else:
                self.show_result(result)
                
        except Exception as e:
            self.show_error(f"Analysis failed: {e}")
    
    def display_image(self, image_path):
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            image.thumbnail((400, 400))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not display image: {e}")
    
    def show_result(self, result):
        prediction = result['prediction']
        confidence = result['confidence']
        fake_prob = result['fake_probability']
        
        if prediction == 'FAKE':
            color = '#e74c3c'
            icon = '❌ FAKE CONTENT DETECTED'
            message = "This image appears to be artificially generated!"
        else:
            color = '#2ecc71'
            icon = '✅ REAL CONTENT'
            message = "This image appears to be authentic!"
        
        self.result_label.config(text=icon, fg=color)
        self.confidence_label.config(
            text=f"Confidence: {confidence:.1%} | Fake Probability: {fake_prob:.3f}\n{message}"
        )
        
        # Show detailed info in messagebox
        details = f"""
Analysis Results:

🎯 Prediction: {prediction}
📊 Confidence: {confidence:.1%}
🔍 Fake Probability: {fake_prob:.3f}
🔍 Real Probability: {result['real_probability']:.3f}

{message}
"""
        if confidence < 0.7:
            details += "\n⚠️ Note: Low confidence - result may be less reliable"
        
        messagebox.showinfo("Analysis Details", details)
    
    def show_error(self, error_message):
        self.result_label.config(text="❌ Analysis Error", fg='#e74c3c')
        self.confidence_label.config(text=error_message)
        messagebox.showerror("Error", error_message)

def main():
    root = tk.Tk()
    app = DeepFakeDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()