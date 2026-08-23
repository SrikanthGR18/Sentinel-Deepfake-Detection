from dataset import DeepfakeDataset

train = DeepfakeDataset("datasets/train")

print(len(train))

img, label = train[0]

print(img.shape)

print(label)