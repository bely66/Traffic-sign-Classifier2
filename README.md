# Traffic-sign-Classifier

## Dataset :

the [German Traffic Sign Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/) on kaggle which contains 39k training images and 12k test images 

## Code : 
__models/trrafic_nn.py__  contains different models to try on the dataset 

## Training : 
```
!python train.py --dataset data \
	--model output/trafficsignnet.model --plot output/plot.png
```

## Predict :
```
!python predict.py --model output/trafficsignnet.model \
	--images data/Test \
	--examples examples
```
