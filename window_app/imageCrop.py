from PIL import Image
import os.path

def center_crop_4_5(image_path, output_path):
    try:
        # Open the image
        img = Image.open(image_path)
        width, height = img.size
        print(f"Original dimensions: {width}x{height}")

        # Calculate new dimensions (4/5 or 80% of original)
        new_width = int(width * 0.7)
        new_height = int(height * 0.7)

        # Calculate the coordinates for the center crop
        # The box tuple is (left, upper, right, lower)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        # Ensure coordinates are integers
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)

        # Perform the crop
        cropped_img = img.crop((left, top, right, bottom))
        print(f"Cropped dimensions: {cropped_img.size[0]}x{cropped_img.size[1]}")

        # Save the new image
        cropped_img.save(output_path)
        print(f"Image successfully cropped and saved to {output_path}")
        
        # Optional: display the cropped image
        # cropped_img.show() 

    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# Replace 'original_image.jpg' with your image file's name
# Replace 'cropped_image.jpg' with the desired output name
center_crop_4_5('crop_002_4.jpg', 'crop_again_002_4.jpg')
