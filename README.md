# ImgGen
Image generation from scene graphs

## Authors
- Gábor Kálmár (XCCGBS)
- Márton Tárnok (GGDVB2)
- Domonkos Kostyál (O32YUM)

## compiling the dataset
- We downloaded the **COCO** train database from these links:
  - The images:       http://images.cocodataset.org/zips/train2017.zip
  - The annotations (relationships, object,  masks, etc.):  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- From almost 20 GB of images we sorted 10000 pieces for training, validating and testing.
- We used a 10% testing and a 10% validation split.
- We cropped and resized the images to 64x64 to reduce computational time.
- Fortunately the **COCO** database is well prepared and we have to sort out a few things only:
  - Minimum 1 object has to be on the image
  - Maximum 4 objects have to be on the image
  - If the object mask is too small (its smaller than the original picture times 0.02) then sort It out

### CocoSceneGraphDataset class
In order to make the loading of the pictures straightforward, we made an individual class for this purpose.
