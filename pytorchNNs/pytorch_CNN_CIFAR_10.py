import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
"""_summary_
    Simple pytorch CNN that gets a final loss of 0.326 and acc of 92.8% on categorizing CIFAR-10
    
"""

#transform the training data with randomization to improve generalization
transf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
#test transform no randomization
test_transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#load training data with transformation
total_data = torchvision.datasets.CIFAR10(root = "./data", transform = transf, train = True, download= True)

#select a random 90% as training 10% for validation
train_data, val_data = torch.utils.data.random_split(total_data, [45000,5000])

train_load = torch.utils.data.DataLoader(train_data, 256, True, num_workers=8)

val_loader = torch.utils.data.DataLoader(val_data, 256, False, num_workers=8)

#load test data with normal transformation
test_data = torchvision.datasets.CIFAR10(root = "./data", transform = test_transf, train = False, download= True)

test_load = torch.utils.data.DataLoader(test_data, 256, False, num_workers=8)


# model is 6 Conv2d layers with ReLU activation function, MaxPool2d, and Batchnorm followed by
# 2 fc layers with batchnorm and dropout
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 10)
            )
        
    def forward(self, x):
        return self.net(x)
    


 
model = myCNN()

#use gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = model.to(device)



#100 time steps
Epochs = 100
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=0.01, steps_per_epoch=len(train_load), epochs=Epochs
)

#train CNN
for epoch in range(Epochs):
    running_loss = 0.
    #train
    model.train()
    for images, labels in train_load:
        images, labels = images.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
    #validate
    model.eval()
    val_running_loss = 0
    val_running_correct = 0
    with torch.no_grad():
        for images,labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_running_correct += (preds == labels).sum().item()
    
    epoch_loss = running_loss / len(train_load.dataset)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Val_loss: {val_running_loss/len(val_loader.dataset): .4f}, Val_acc: {val_running_correct/len(val_loader.dataset): .4f}")

model.eval()

#test CNN
running_loss = 0
running_correct =0
with torch.no_grad():
    for images, labels in test_load:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        

epoch_loss = running_loss / len(test_load.dataset)
print(f"Final Loss: {epoch_loss:.4f}, Final Acc: {running_correct/len(test_load.dataset): .4f}")


# Output:
#     Using device: cuda
# Epoch 1, Loss: 1.5154, Val_loss:  1.2747, Val_acc:  0.5446
# Epoch 2, Loss: 1.1011, Val_loss:  1.0754, Val_acc:  0.6286
# Epoch 3, Loss: 0.9328, Val_loss:  1.1340, Val_acc:  0.6060
# Epoch 4, Loss: 0.8280, Val_loss:  1.0399, Val_acc:  0.6444
# Epoch 5, Loss: 0.7689, Val_loss:  0.8834, Val_acc:  0.6930
# Epoch 6, Loss: 0.7272, Val_loss:  0.8687, Val_acc:  0.7030
# Epoch 7, Loss: 0.6882, Val_loss:  0.7968, Val_acc:  0.7246
# Epoch 8, Loss: 0.6554, Val_loss:  0.8171, Val_acc:  0.7242
# Epoch 9, Loss: 0.6301, Val_loss:  0.7575, Val_acc:  0.7422
# Epoch 10, Loss: 0.6092, Val_loss:  0.7004, Val_acc:  0.7544
# Epoch 11, Loss: 0.5808, Val_loss:  0.7616, Val_acc:  0.7482
# Epoch 12, Loss: 0.5520, Val_loss:  0.6597, Val_acc:  0.7758
# Epoch 13, Loss: 0.5245, Val_loss:  0.6933, Val_acc:  0.7634
# Epoch 14, Loss: 0.5157, Val_loss:  0.7034, Val_acc:  0.7726
# Epoch 15, Loss: 0.4927, Val_loss:  0.6052, Val_acc:  0.7934
# Epoch 16, Loss: 0.4731, Val_loss:  0.5631, Val_acc:  0.8100
# Epoch 17, Loss: 0.4617, Val_loss:  0.6537, Val_acc:  0.7800
# Epoch 18, Loss: 0.4429, Val_loss:  0.5581, Val_acc:  0.8068
# Epoch 19, Loss: 0.4427, Val_loss:  0.5448, Val_acc:  0.8208
# Epoch 20, Loss: 0.4202, Val_loss:  0.5755, Val_acc:  0.8000
# Epoch 21, Loss: 0.4165, Val_loss:  0.5569, Val_acc:  0.8074
# Epoch 22, Loss: 0.4041, Val_loss:  0.5509, Val_acc:  0.8152
# Epoch 23, Loss: 0.3884, Val_loss:  0.5769, Val_acc:  0.8046
# Epoch 24, Loss: 0.3766, Val_loss:  0.5589, Val_acc:  0.8188
# Epoch 25, Loss: 0.3701, Val_loss:  0.5284, Val_acc:  0.8286
# Epoch 26, Loss: 0.3612, Val_loss:  0.5334, Val_acc:  0.8206
# Epoch 27, Loss: 0.3559, Val_loss:  0.5268, Val_acc:  0.8288
# Epoch 28, Loss: 0.3458, Val_loss:  0.4844, Val_acc:  0.8394
# Epoch 29, Loss: 0.3333, Val_loss:  0.5604, Val_acc:  0.8148
# Epoch 30, Loss: 0.3249, Val_loss:  0.5207, Val_acc:  0.8316
# Epoch 31, Loss: 0.3171, Val_loss:  0.5394, Val_acc:  0.8264
# Epoch 32, Loss: 0.3079, Val_loss:  0.4649, Val_acc:  0.8450
# Epoch 33, Loss: 0.2978, Val_loss:  0.5390, Val_acc:  0.8264
# Epoch 34, Loss: 0.2986, Val_loss:  0.5167, Val_acc:  0.8302
# Epoch 35, Loss: 0.2808, Val_loss:  0.4435, Val_acc:  0.8532
# Epoch 36, Loss: 0.2723, Val_loss:  0.5890, Val_acc:  0.8180
# Epoch 37, Loss: 0.2719, Val_loss:  0.4760, Val_acc:  0.8404
# Epoch 38, Loss: 0.2654, Val_loss:  0.4471, Val_acc:  0.8546
# Epoch 39, Loss: 0.2544, Val_loss:  0.5058, Val_acc:  0.8378
# Epoch 40, Loss: 0.2526, Val_loss:  0.5335, Val_acc:  0.8350
# Epoch 41, Loss: 0.2482, Val_loss:  0.4573, Val_acc:  0.8472
# Epoch 42, Loss: 0.2410, Val_loss:  0.4675, Val_acc:  0.8422
# Epoch 43, Loss: 0.2317, Val_loss:  0.4287, Val_acc:  0.8568
# Epoch 44, Loss: 0.2283, Val_loss:  0.4876, Val_acc:  0.8412
# Epoch 45, Loss: 0.2217, Val_loss:  0.4592, Val_acc:  0.8526
# Epoch 46, Loss: 0.2152, Val_loss:  0.4398, Val_acc:  0.8602
# Epoch 47, Loss: 0.2098, Val_loss:  0.4493, Val_acc:  0.8628
# Epoch 48, Loss: 0.2025, Val_loss:  0.4383, Val_acc:  0.8652
# Epoch 49, Loss: 0.1940, Val_loss:  0.4135, Val_acc:  0.8720
# Epoch 50, Loss: 0.1867, Val_loss:  0.4664, Val_acc:  0.8546
# Epoch 51, Loss: 0.1846, Val_loss:  0.3999, Val_acc:  0.8736
# Epoch 52, Loss: 0.1798, Val_loss:  0.4252, Val_acc:  0.8662
# Epoch 53, Loss: 0.1745, Val_loss:  0.4211, Val_acc:  0.8730
# Epoch 54, Loss: 0.1697, Val_loss:  0.4191, Val_acc:  0.8696
# Epoch 55, Loss: 0.1613, Val_loss:  0.3868, Val_acc:  0.8802
# Epoch 56, Loss: 0.1588, Val_loss:  0.4087, Val_acc:  0.8692
# Epoch 57, Loss: 0.1543, Val_loss:  0.4077, Val_acc:  0.8706
# Epoch 58, Loss: 0.1444, Val_loss:  0.4649, Val_acc:  0.8634
# Epoch 59, Loss: 0.1393, Val_loss:  0.4457, Val_acc:  0.8718
# Epoch 60, Loss: 0.1389, Val_loss:  0.4092, Val_acc:  0.8780
# Epoch 61, Loss: 0.1266, Val_loss:  0.4364, Val_acc:  0.8752
# Epoch 62, Loss: 0.1276, Val_loss:  0.4052, Val_acc:  0.8818
# Epoch 63, Loss: 0.1243, Val_loss:  0.3711, Val_acc:  0.8862
# Epoch 64, Loss: 0.1125, Val_loss:  0.4151, Val_acc:  0.8814
# Epoch 65, Loss: 0.1121, Val_loss:  0.4072, Val_acc:  0.8866
# Epoch 66, Loss: 0.1072, Val_loss:  0.4159, Val_acc:  0.8842
# Epoch 67, Loss: 0.1024, Val_loss:  0.4013, Val_acc:  0.8846
# Epoch 68, Loss: 0.0975, Val_loss:  0.4124, Val_acc:  0.8894
# Epoch 69, Loss: 0.0943, Val_loss:  0.4200, Val_acc:  0.8824
# Epoch 70, Loss: 0.0889, Val_loss:  0.3791, Val_acc:  0.8978
# Epoch 71, Loss: 0.0871, Val_loss:  0.4074, Val_acc:  0.8920
# Epoch 72, Loss: 0.0800, Val_loss:  0.3717, Val_acc:  0.8956
# Epoch 73, Loss: 0.0798, Val_loss:  0.3814, Val_acc:  0.8968
# Epoch 74, Loss: 0.0703, Val_loss:  0.3723, Val_acc:  0.9000
# Epoch 75, Loss: 0.0682, Val_loss:  0.4250, Val_acc:  0.8896
# Epoch 76, Loss: 0.0652, Val_loss:  0.4119, Val_acc:  0.8898
# Epoch 77, Loss: 0.0609, Val_loss:  0.4067, Val_acc:  0.8952
# Epoch 78, Loss: 0.0607, Val_loss:  0.4137, Val_acc:  0.8946
# Epoch 79, Loss: 0.0549, Val_loss:  0.4287, Val_acc:  0.8940
# Epoch 80, Loss: 0.0564, Val_loss:  0.3940, Val_acc:  0.8978
# Epoch 81, Loss: 0.0521, Val_loss:  0.4260, Val_acc:  0.8946
# Epoch 82, Loss: 0.0488, Val_loss:  0.4044, Val_acc:  0.9008
# Epoch 83, Loss: 0.0450, Val_loss:  0.3999, Val_acc:  0.9022
# Epoch 84, Loss: 0.0471, Val_loss:  0.4115, Val_acc:  0.9018
# Epoch 85, Loss: 0.0480, Val_loss:  0.4015, Val_acc:  0.9006
# Epoch 86, Loss: 0.0434, Val_loss:  0.3948, Val_acc:  0.9018
# Epoch 87, Loss: 0.0402, Val_loss:  0.3829, Val_acc:  0.9054
# Epoch 88, Loss: 0.0380, Val_loss:  0.4220, Val_acc:  0.8978
# Epoch 89, Loss: 0.0344, Val_loss:  0.4139, Val_acc:  0.9024
# Epoch 90, Loss: 0.0365, Val_loss:  0.4187, Val_acc:  0.9036
# Epoch 91, Loss: 0.0347, Val_loss:  0.3989, Val_acc:  0.9044
# Epoch 92, Loss: 0.0364, Val_loss:  0.3983, Val_acc:  0.9022
# Epoch 93, Loss: 0.0351, Val_loss:  0.4113, Val_acc:  0.9044
# Epoch 94, Loss: 0.0331, Val_loss:  0.4172, Val_acc:  0.8988
# Epoch 95, Loss: 0.0308, Val_loss:  0.3830, Val_acc:  0.9052
# Epoch 96, Loss: 0.0312, Val_loss:  0.3964, Val_acc:  0.9048
# Epoch 97, Loss: 0.0338, Val_loss:  0.4121, Val_acc:  0.9052
# Epoch 98, Loss: 0.0322, Val_loss:  0.4090, Val_acc:  0.9056
# Epoch 99, Loss: 0.0312, Val_loss:  0.4049, Val_acc:  0.9102
# Epoch 100, Loss: 0.0333, Val_loss:  0.3936, Val_acc:  0.9094
# Final Loss: 0.3260, Final Acc:  0.9284