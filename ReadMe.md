# TODO

tnie prescribed these on 2024/09/19 :)

- [x] Refactor `TrainDataset` and `TestDataset` into a single, unified class. Make `root_dir` a positional parameter to the `__init__` method.
  - We want to reduce code duplication as much as possible. Duplication leads to more bugs.
- [ ] Fix your PyCharm's intellisense! Tell PyCharm which Python executable you're using. Make red and green squiggles go away.
- [ ] Notice in `TrainDataset`'s `__init__` method, you are manually padding/resizing the images and min-max normalizing them. Prefer to use a transformation pipeline, like the one provided in PyTorch.
  - Refer to [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) to learn how to use `torchvision.transforms` to build a pipeline, and rewrite your resize and min-max normalization code into this pipeline (super easy).
  - One major advantage of using this transformation is that it's much easier to read and easier to reverse. For example, you might want to predict the segmentation mask of a new image with your trained model, and want to reverse transform the predicted mask into the orignal size (and not 128x128).

# Predicting masks using UNet

1. Data Preparation
   * Create Two classes of Dataloader: "trainloader", "testloader," each using a 3:1 ratio referring to the batch.csv file under "/Users/luozisheng/Documents/Zhu‘s Lab/PyTorch_Lightening"
2. Create a standard OOP file for PyTorch Lightening
3. Connect PyTorch Lightening to WandB and log information online