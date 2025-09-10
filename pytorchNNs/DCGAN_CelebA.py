import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
"""
    Failed attempt at DCGAN to generate faces by training on CelebA dataset, dropped because testing and training for 
    tuning hyperparameters takes too long
"""
img_size = 64
bat_size = 128
nc = 3
lat = 100
nf = 64
epochs = 10
ler = 0.0002
beta1 = 0.5


transf = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))    
])

data = torchvision.datasets.CelebA(
    root="./data", split="train", download=True, transform=transf
)

dataload = torch.utils.data.DataLoader(data, bat_size, True, num_workers=4)

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d( lat, nf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.ConvTranspose2d( nf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )
    def forward(self, X):
        return self.net(X)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, X):
        return self.net(X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

gen = Generator().to(device)
dis = Discriminator().to(device)

gen.apply(weights_init)
dis.apply(weights_init)

criterion = torch.nn.BCELoss()
gen_optim = torch.optim.Adam(gen.parameters(), lr = ler, betas = (beta1,0.999))
dis_optim = torch.optim.Adam(dis.parameters(), lr = ler, betas = (beta1,0.999))

gen.train()
dis.train()
for epoch in range(epochs):
    totalDRLoss = 0
    totalDFLoss = 0
    totalGLoss = 0
    for i, (images, _) in enumerate(dataload):
        
        b_size = images.size(0)
        #discriminator real
        dis.zero_grad()
        gen.zero_grad()

        images = images.to(device)
        labels = torch.full((b_size,), .95, device= device)
        outputs = dis(images).view(-1)
        lossD_real = criterion(outputs, labels)
        lossD_real.backward()
        totalDRLoss += lossD_real.item()
        
        #discriminator fake
        noise = torch.randn(b_size, lat, 1, 1, device = device)
        fakes = gen(noise)
        labels.fill_(0.05)
        outputs = dis(fakes.detach()).view(-1)
        lossD_fake = criterion(outputs, labels)
        lossD_fake.backward()
        totalDFLoss += lossD_fake.item()
        dis_optim.step()
        
        #gen

        labels.fill_(.95)
        output = dis(fakes).view(-1)
        loss = criterion(output, labels)
        loss.backward()
        totalGLoss += loss.item()
        gen_optim.step()
        if i==len(dataload)-1:
            print(f"Epoch: {epoch}, LossDR :{totalDRLoss/len(dataload): .4f}, LossDF: {totalDFLoss/len(dataload): .4f}, LossG: {totalGLoss/len(dataload): .4f}")


grid_size = 8
gen.eval()
noise = torch.randn(grid_size**2,lat,1,1,device=device)

#generate, make grid, save, and plot image
with torch.no_grad():
    fake = gen(noise)

grid = vutils.make_grid(fake, padding=2, normalize=True)

plt.figure(figsize = (8,8))
plt.axis("off")
plt.title("DCGAN Generated Images")
#permute to change from C,H,W to H,W,C for matplot
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
plt.savefig("dcgan_grid.png")

        

