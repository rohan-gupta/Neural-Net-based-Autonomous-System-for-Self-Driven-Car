# Neural-Net-based-Autonomous-System-for-Self-Driven-Car

![Alt text](https://github.com/rohan-gupta/Neural-Net-based-Autonomous-System-for-Self-Driven-Car/blob/master/banner.png)

(Screenshot from trained model visualiser - black dot representing the predicted steer angle / white dot representing the actual steer angle)

## Objective

To train a Convolutional Neural Network (CNN) for enabling a self-driving car. 

## Project Description

Transfer learning is a way to utilize previously trained neural nets. We will use ResNet 50 since it is a high quality Neural Net for object recognition that has been pre-trained on a very large dataset.

The ResNet 50 layer is followed by 3 fully connected layers having 512, 256 and 64 neurons respectively.

We are using Adam optimizer to minimize loss and the learning rate is set at 0.001.

## Demonstration Video

Link - https://drive.google.com/file/d/1yj4Ldpktk51XuPMde0u8QrYMVa6qOMOO/view?usp=share_link

## Instructions

run the following commands in terminal

- `pip install -r requirements.txt`
- `cd src`
- `python main.py` or `python visualize.py`
