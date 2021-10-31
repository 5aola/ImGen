# ImGen
Image generation from scene graphs

## Authors
- Gábor Kalmár (XCCGBS)
- Márton Tárnok (GGDVB2)
- Domonkos Kostyál (O32YUM)

## compiling the dataset
- We downloaded the **COCO** train database from these links:
  - The images:       http://images.cocodataset.org/zips/train2017.zip
  - The annotations (relationships, object,  masks, etc.):  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- From almost 20 GB of images we sorted 10000 pieces for training, validating and testing.
- We will use a 10% testing and a 10% validation split.
- We cropped and resized the images to 64x64 to reduce computational time.
- Fortunately the **COCO** database is well prepared and we have to sort out a few things only:
  - Minimum 1 object has to be on the image
  - Maximum 4 objects can be on the image
  - If the object mask is too small (its smaller than the original picture times 0.02) then sort It out
- The data compiling is in [SceneGraphGeneration.ipynb](/SceneGraphGeneration.ipynb)

### CocoSceneGraphDataset class
- In order to make the loading of the pictures and annotations straightforward, we made an individual class for this purpose.
- We transformed this class from the [**sg2im**](https://github.com/google/sg2im) project that also uses and processes the **COCO** dataset.
