# SemanticSegmentation for aerial images taken by drone

Dataset: https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

I used the Unet model from here(https://github.com/qubvel/segmentation_models.pytorch).

I'll show both best prediction and worst predction and also the reason why this happens.
<h1>Best prediction</h1>
<p float="left">
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/best3/origin_best16.png" width="320" />
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/best3/mask_best16.png" width="320" /> 
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/best3/pred_best16.png" width="320" />
</p>



<h1>Worst prediction</h1>
<p float="left">
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/worst3/origin_worst5.png" width="320" />
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/worst3/mask_worst5.png" width="320" /> 
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/worst3/pred_worst5.png" width="320" />
</p>

<h1>Analytics</h1>
These graph shows distribution of the number of times that left sides color pixels show in the train dataset.
<p float="left">
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/imcolor_dist.png" width="320" />
  <img src="https://github.com/O-suke12/SemanticSegmentation/blob/master/evaluation/image_dist.png" width="320" />
</p>
As you can see, this train dataset has bias. But it's pretty normal phenomenon. 
Because the size of area(pixels) of people or bikes or other some stuffs are really small from above.
So it's difficult for the model to fit to those tiny objects.

