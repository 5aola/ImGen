# ImGen!
 ### Documentation
 [Documentation](/document.pdf) of the project.
  
Image generation from scene graphs.
The goal of the project was to gain a deeper 
understanding of graph convolutional networks and 
image generation using scene graphs. A complex sentence 
can often be more explicitly represented as a scene 
graph of objects and their relationships. Scene 
graphs are a powerful structured representation for 
both images and language. 

## Authors
- Gábor Kalmár (XCCGBS)
- Márton Tárnok (GGDVB2)
- Domonkos Kostyál (O32YUM)

### Our generated pictures from graphs:

  Three people                                                                                                                                         |  Two giraffes
:-----------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------:
<img  alt="image" src="https://user-images.githubusercontent.com/56648499/146053876-7c78a35a-7956-4abf-86c6-5e9d829953b3.png" width="150" height="150">|<img alt= "image" src="https://user-images.githubusercontent.com/56648499/146054309-ce476077-e339-4c27-98c7-a61c40b8ae3f.png" width="150" height="150">

## Improvement during training

At the beginning without standardization:

![Screenshot 2021-12-11 163313](https://user-images.githubusercontent.com/56648499/146056827-b6ed15b8-5e47-438b-b710-f7eb33bd2e57.png)

After standardization:
![Screenshot 2021-12-12 131219](https://user-images.githubusercontent.com/56648499/146057290-81a56786-cbcf-4eba-bbb1-e808a3f48d43.png)
![Screenshot 2021-12-12 130548](https://user-images.githubusercontent.com/56648499/146057299-3c38da1e-e509-4a32-9241-b22bf4fdfc75.png)
After 30 epochs:
![Screenshot 2021-12-12 204836](https://user-images.githubusercontent.com/56648499/146057358-7dfe9558-bc75-4041-99d9-14beb530eb46.png)


## Compiling the dataset
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

## Second milestone
We decided to change up for PyCharm IDE instead of Jupyter, because as the project got bigger we needed better structure and intellisense.

### Graph Convolutional Networks (GCN)
 - To convert graphs into train data we used graph convolutional network
 - The goal of the this network to convert the graph to a vector, that represents the relation (edges and vertices) of the original scene graph
 - This network takes a object_vector, predicat_vectors and edges
 - We use Embedding network to convert objects and predicates into vectors(dim is 64)
 - The edges are vectors that represent the relationship
 - This model returns with object and predicate that contains the knowledge of their own relations

### Box and Masknets
  - These two nets are trained at the same time
  - Both networks are using object_vectors (the output of the GCN) for training
  - Boxnet is an MLP (Sequential model of linear, batchnorm1D and dropout layers usimg Relu Activation)
  - Masknet is a CNN (Sequential model of UpSample, BatchNorm2D, Conv2D, and ReLU layers)
  - Outputs are box_predicates and mask_predicates

### Training our model
  - We trained the model on NVIDIA 1060 GPU with cuda for a whole night but due to lack of computational resource, we couldn't build an accurate model
  - The imporvement was visible, but there is plenty of  room for improvement with parameter and code optimalization

### Run the model
In order to generate images from scene graphs, follow the next steps:
  - download our pretrained model weights from : https://www.dropbox.com/s/zihnz2cuj009uuy/model.pt?dl=0
  - instert the model.pt file into the NagyHF folder
  - install all the required libraries
  - in order to write own scene graphs use the scene_graphs/inputSceneGraphs.json file and follow the structure
  - we only trained for person and giraffe objects
  - run the run.py program
  - the generated images are visible in the generatedImages folder
  



