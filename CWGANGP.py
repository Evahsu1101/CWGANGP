# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:02:46 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 06:01:38 2024

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 00:55:14 2023

@author: User
"""

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import os
# import pandas as pd  
# import numpy as np  
# import cv2  

# from scipy.stats import entropy
# from torchvision.models import inception_v3
# from torchvision.transforms import ToPILImage
# from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

 
    
    
# 數據集定義
class CustomDataset(Dataset):
    # 初始化
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform

        self.image_files = []
        self.labels = []

        # 加載數據和標籤
        for label in os.listdir(root_folder):
            label_path = os.path.join(root_folder, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    self.image_files.append(os.path.join(label_path, image_file))
                    self.labels.append(int(label))  # 文件夾名稱為0 OR 1

    # 獲取數據及長度
    def __len__(self):
        return len(self.image_files)

    # 獲取數據項
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# 圖像預處理256*256
transform = transforms.Compose([
    transforms.Resize((512, 512)),  
    transforms.ToTensor(), 
])


# 定義路徑
# root_folder = "C:\\Users\\User\\Desktop\\分類"
# output_folder = r'C:\Users\User\Desktop\Label Vector datasets\C\CWGAN-GP\測試生成'
# model_save_dir = r'C:\Users\User\Desktop\Label Vector datasets\C\CWGAN-GP\model'
root_folder = r"./classification"
output_folder = r"./generation_test"
model_save_dir = r"./model"
custom_dataset = CustomDataset(root_folder, transform=transform)

# dataloader
batch_size = 6
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)


# 定義生成器網絡
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, img_channels, img_size):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # 嵌入層將類別條件轉換為條件向量

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, conditions):
        # 將類別條件轉換為條件向量

        # 將隨機噪聲和條件連接起來
        x = torch.cat((noise, conditions), dim=1)
        x = self.fc(x)
        # 重新調整輸出為圖像大小
        x = x.view(x.size(0), self.img_channels, self.img_size, self.img_size)
        return x

# 定義判別器網絡
class Discriminator(nn.Module):
    def __init__(self, condition_dim, img_channels, img_size):
        super(Discriminator, self).__init__()

        self.condition_dim = condition_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # 嵌入層將類別條件轉換為條件向量

        self.fc = nn.Sequential(
            nn.Linear(img_channels * img_size * img_size + condition_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, img, conditions):
        # 將類別條件轉換為條件向量


        # 將圖像和條件連接起來
        x = img.view(img.size(0), -1)
        x = torch.cat((x, conditions), dim=1)
        x = self.fc(x)
        return x


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    



# 超參數設置
latent_size = 200
latent_dim = 200
condition_dim = 1
img_channels = 3
img_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 創建及初始化生成器及判別器
generator = Generator(latent_dim, condition_dim, img_channels, img_size).to(device)
discriminator = Discriminator(condition_dim, img_channels, img_size).to(device)

# 定義損失函數和優化器

import torch.optim as optim

# 初始化生成器和判别器的优化器，加入weight decay
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))




def plot_generator_loss(generator_losses, title):
    plt.figure(figsize=(12, 8))
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def wasserstein_loss(output, target):
    return torch.mean(output * target)

def compute_gradient_penalty(D, real_samples, fake_samples, conditions, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates = interpolates.to(device).requires_grad_(True)
    d_interpolates = D(interpolates, conditions)
    fake = torch.ones(real_samples.size(0), 1, device=device, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

    
    
    
# 初始化紀錄的列表
overall_accuracies = []
class_0_accuracies = []
class_1_accuracies = []
generator_losses = []
discriminator_losses = []
is_scores_list = []  
fid_scores_list = [] 
lpips_scores = []  

#訓練參數
# 在訓練迴圈之前創建損失列表
generator_losses = []
discriminator_losses = []
input_dim = 200
epochs = 5000
lambda_gp = 10  
n_critic = 2
sample_interval = 10
img_size = 256
batch_size = 43
num_classes = 2
epochs = 10000
sample_interval = 100  # 10個EPOCH生成一張圖
num_images_to_generate = 100
sample_interval1 = 100
#訓練迴圈
for epoch in range(epochs):
    for batch_idx, (images, labels) in enumerate(data_loader):
        real_images = images.to(device)

        conditions = labels.to(device)
        conditions = labels.unsqueeze(1).to(device)


        # Training discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = generator(z, conditions)
        real_loss = -torch.mean(discriminator(real_images, conditions))
        fake_loss = torch.mean(discriminator(fake_images.detach(), conditions))
        gradient_penalty = compute_gradient_penalty(discriminator, real_images, fake_images, conditions, device)
        d_loss = real_loss + fake_loss + lambda_gp * gradient_penalty
        d_loss.backward()
        optimizer_D.step()

        # Train generator every n_critic steps
        if batch_idx % n_critic == 0:
            optimizer_G.zero_grad()
            generated_images = generator(z, conditions)
            g_loss = -torch.mean(discriminator(generated_images, conditions))
            g_loss.backward()
            optimizer_G.step()
            generator_losses.append(g_loss.item())
            discriminator_losses.append(d_loss.item())

        # Print training update
        if batch_idx % sample_interval == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(data_loader)}], '
                  f'Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}')


        # 保存生成器的生成的圖像和標籤
        if epoch % sample_interval == 0:
            with torch.no_grad():
             # 將生成的圖像和標籤丟入變量中
               samples_with_labels = [{'image': fake_images[i], 'label': conditions[i].cpu().numpy()} for i in range(batch_size)]
               save_image(fake_images.data[:1], f"{output_folder}/Test2epoch_{epoch}.png", nrow=1, normalize=True)
               
 # 每1000个Epoch保存一次模型
        if (epoch + 1) % 1000 == 0:
            generator_save_path = os.path.join(model_save_dir, f'CWGAN-GP C T2 generator_epoch_{epoch+1}.pth')
            discriminator_save_path = os.path.join(model_save_dir, f'CWGAN-GP C T2 discriminator_epoch_{epoch+1}.pth')
        
            torch.save(generator.state_dict(), generator_save_path)
            torch.save(discriminator.state_dict(), discriminator_save_path)
            
            print(f"Models saved at epoch {epoch+1}")



# 定义保存模型的文件夹
save_folder = r"./model"
os.makedirs(save_folder, exist_ok=True)  # 确保文件夹存在

# 定义保存的文件路径
generator_path = os.path.join(save_folder, 'CWGAN-GP T2 generator EPOCH10000.pth')
discriminator_path = os.path.join(save_folder, 'CWGAN-GP T2 discriminator EPOCH10000.pth')

# 保存模型
torch.save(generator.state_dict(), generator_path)
torch.save(discriminator.state_dict(), discriminator_path)

print(f"Generator model saved at {generator_path}")
print(f"Discriminator model saved at {discriminator_path}")

              
                
#訓練結束畫圖      
plot_generator_loss(generator_losses, 'CWGAN-GP T2 Training Losses')


# 保存模型參數
torch.save(generator.state_dict(), 'CWGAN-GP T2 generatorEPOCH10000.pth')
torch.save(discriminator.state_dict(), 'CWGAN-GP T2 discriminatorEPOCH10000.pth')



import os
import torch
from torchvision.utils import save_image

def generate_and_save_images(generator, category, num_images, latent_dim, output_dir, device):

    os.makedirs(output_dir, exist_ok=True)
    # 設置生成器為評估模式
    generator.eval()

    noise = torch.randn(num_images, latent_dim, device=device)

    labels = torch.full((num_images, 1), category, device=device, dtype=torch.float32)  # 保持与训练时的标签维度一致
    # 生成圖片
    with torch.no_grad():
        generated_images = generator(noise, labels).detach().cpu()
    # 保存圖片
    for i, img in enumerate(generated_images):
        save_image(img, os.path.join(output_dir, f'category_{category}_image_{i}.png'))

latent_dim = 200 

# output_dir = 'C:\\Users\\User\\Desktop\\生成4'
output_dir = r"./generation4"
# 设置设备（例如 "cpu" 或 "cuda"）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载生成器模型
generator = Generator(latent_dim, condition_dim=1, img_channels=3, img_size=512).to(device)
generator.load_state_dict(torch.load('CWGAN-GP generatorEPOCH3000.pth', map_location=device))

# 生成类别0的图片
generate_and_save_images(generator, category=0, num_images=100, latent_dim=latent_dim, output_dir=output_dir, device=device)
# 生成类别1的图片
generate_and_save_images(generator, category=1, num_images=100, latent_dim=latent_dim, output_dir=output_dir, device=device)
