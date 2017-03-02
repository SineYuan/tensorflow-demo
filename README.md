# tensorflow-demo

a tensorflow serving Go client demo

## Prerequirements

 + tensorflow
 + Go 1.7 at lease
 
## Installation
 
 ```
 cd $GOPATH/src
 git clone https://github.com/SineYuan/tensorflow-demo.git
 ```
 
## Mnist digit classification
 
### model training
 
```
 cd $GOPATH/src/tensorflow-demo/mnist
 
// single layer perceptron with softmax
python mnist_softmax.py --model_version=1
 
//convolutional neural networks
python mnist_cnn.py --model_version=2
```

### run tensorflow model server in docker
```
docker run -p 8500:8500 -v $PWD:/work sineyuan/tensorflow_model_server --model_base_path=/work/model
```

### run api server
```
cd ..
go run main.go
``` 

view demo at [http://localhost:1323](http://localhost:1323)