import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import Gmodel
from os.path import join, dirname, abspath
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Gmodel.GModel().to(device)

current_dir = dirname(abspath(__file__))

model_path = join(current_dir, 'Models/GANRemoveBackground.pth')

generator.load_state_dict(torch.load(model_path, map_location=device))

generator.eval()

def convert_to_png(img):
    if img.format != 'PNG':
        img = img.convert('RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return Image.open(buffer)
    else:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        return img



transform = transforms.Compose([
    transforms.ToTensor()
])

def generate_image(img):
    input_img = convert_to_png(img)
    
    input_img = transform(input_img).unsqueeze(0)

    with torch.no_grad():
        input_img = input_img.to(device)
        nobg_img = generator(input_img)
        nobg_img = nobg_img.squeeze().cpu()

    nobg_img = transforms.ToPILImage()(nobg_img)

    return nobg_img

# if __name__ == "__main__":
    # test_image_path = 'D:/Downloads/images.png'
    # test_image = Image.open(test_image_path)
    # converted_image = convert_to_png(test_image)
    # converted_image = transform(converted_image).unsqueeze(0)
    # print(converted_image.shape)


# import torch
# import Gmodel

# checkpoint_path = "D:/Desktop/Python_pj/AI_Assistant/Models/last_checkpoint-reduce_Lr.pth"
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# generator = Gmodel.GModel()
# generator.load_state_dict(checkpoint['generator'].state_dict)
# new_generator_path = "GANRemoveBackground.pth"
# torch.save(generator.state_dict(), new_generator_path)
# print(f"Generator state_dict has been saved to {new_generator_path}")