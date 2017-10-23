# Neural Network in go (golang)
These are the source files of Version 1.0 of a artifical neural network in the Go programming language (Golang).

# Neuronales Netzwerk in go (golang)
Das ist die erste Version eines Neuronalen Netzwerkes in Go.

# License

This source code is subject to the terms of the Mozilla Public
License, version 2.0 (MPL-2.0). If a copy of the MPL was not
distributed with this software, it is also available online at
<http://mozilla.org/MPL/2.0/>.  For futher information about the MPL see <http://www.mozilla.org/MPL/2.0/FAQ.html>.

For more information about artificial neural network see: 
https://en.wikipedia.org/wiki/Artificial_neural_network
 
# Functions of the own written artifical Neural Network

* This program use only **standard go packages**. 
* In the version 1.0 one can **config one or more neural networks** with InputNodes, HiddenNodes, OutputNodes, Epochs and the LearningRate. 
* There is an **interface** between the **Neural Network** and the **Data Set Port**. The Data Set Port read and prepare the training, validation (not jet used) and test data set. 
* A Data Set Port to **Yann LeCun's MNIST dataset of handwritten digits** is included, but you must download the MNIST dataset (see Yann LeCun's MNIST page: http://yann.lecun.com/exdb/mnist/ ).
* The main program includes a code to get **profile information**.

# Das selbst geschriebene Neuronale Neetzwerk Programm  - Eigenschaften

* Dieses Programm verwendet nur **Standard go Pakete**.
* In der Version 1.0 kann eine **Konfiguration für ein oder mehrere Neuronale Netzwerke** mit den Parametern InputNodes, HiddenNodes, OutputNodes, Epochs und LearningRate erfolgen.
* Es gibt eine **Schnittstelle** zwischen dem **Neuronalen Netz** (Neural Network) und dem **Vorverarbeiten der Daten** (Data Set Port). Der Vorprozess zum Einlesen und Vorverarbeiten der Trainings-, Validierungs- (wird noch nicht verwendet) und Testdatensätze wird so von dem Neuronalen Netz getrennt.
* Eine Implementation dieser Schnittstelle von **Yann LeCun's MNIST datensatz von handgeschriebenen Ziffern** ist enthalten. Die MNIST-Daten müssen heruntergeladen werden (siehe Yann LeCun's MNIST Seite:  http://yann.lecun.com/exdb/mnist/ ).
* Das Hauptprogramm enthält einen Code zum Schreiben von **Profile-Informationen**. 


# The Neural Network

```go
type NNconfig struct { // to configurate a Neural Network
	InputNodes  int
	HiddenNodes int
	OutputNodes int
	Epochs      int
	LearningRate float64
}

func (nn *NeuralNetwork) InitNeuralNetwork(nnConfig NNconfig)
func (nn *NeuralNetwork) Train(ds DataSetInterface, epochs int, learningRate float64) (err error) 
func (nn *NeuralNetwork) Test(ds DataSetInterface) (result float64, err error) 
```

# The Data Set Port - Interface
```go
type DSconfig struct { // to configurate a Data Set
	PathName           string
	TrainingFileName   string
	ValidationFileName string
	TestFileName       string
	InputData          int
	OutputData         int
}

type DataSetInterface interface {
	InitDataSet(DSP.DSconfig)
	OpenTrainingDataFile() error
	OpenValidationDataFile() error
	OpenTestDataFile() error
	CloseDataFile()
	ReadNextDataSet() error
	InputData2InputNodes(inputNodesVector []float64)
	OutputData2TargetNodes(targetNodesVector []float64)
	OutputNotes2OutputData(outputNodesVector []float64)
	AreOutputNotesOK(outputNodesVector []float64) bool
	PrintOutputData()
}
```
# An Example
```go
import (
    // .... and 
	DSP "github.com/hfried/GoNN/dataset_port"
)

var mnistNNconfig NNconfig = NNconfig{InputNodes: 784, HiddenNodes: 200, OutputNodes: 10, Epochs: 1, LerningRate: 0.2}

var mnistDSconfig DSP.DSconfig = DSP.DSconfig{InputData: 784, OutputData: 1, PathName: "mnist_dataset/", TrainingFileName: "mnist_train.csv",
	ValidationFileName: "", TestFileName: "mnist_test.csv"}

func trainAndTest(nnConfig NNconfig, dsConfig DSP.DSconfig) (error, float64) {

	nn := new(NeuralNetwork)
	nn.InitNeuralNetwork(nnConfig)

	ds := new(DSP.MNIST_DataSet)
	ds.InitDataSet(dsConfig)

	err := nn.Train(ds, nn.epochs, nn.learningRate)
	if err != nil {
		return err, 0.0
	}
	result, err := nn.Test(ds)
	if err != nil {
		return err, 0.0
	}
	
	return nil, result
}

func main() {

    startTime := time.Now()
	jahr, mon, tag := startTime.Date()
	stunde, minute, sekunde := startTime.Clock()
	fmt.Printf("Start: %02d.%02d.%d  %02d:%02d:%02d with %d Epochs\n", tag, mon, jahr, stunde, minute, sekunde, mnistNNconfig.Epochs)

	err, result := trainAndTest(mnistNNconfig, mnistDSconfig)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Hit rate: %.2f%% \nused time: %s\n", result*100.0, time.Now().Sub(startTime))
}
```

# To get Profile Information

(see: https://golang.org/pkg/runtime/pprof/  and https://github.com/google/pprof/blob/master/doc/pprof.md )

```go
func main() {
	// CPU-Profile
	f, ferr := os.Create("cpuprofile.prof")
	if ferr != nil {
		log.Fatal("could not create CPU profile: ", ferr)
	}
	if perr := pprof.StartCPUProfile(f); perr != nil {
		log.Fatal("could not start CPU profile: ", perr)
	}
	defer pprof.StopCPUProfile()
	// END CPU-Profile
	
	// ... main-body
	
	// MEM-Profile
	memf, memerr := os.Create("memprofile.prof")
	if memerr != nil {
		log.Fatal("could not create memory profile: ", memerr)
	}
	runtime.GC() // get up-to-date statistics
	if memerr := pprof.WriteHeapProfile(memf); memerr != nil {
		log.Fatal("could not write memory profile: ", memerr)
	}
	memf.Close()
}
```
To read the profile informaion generate pdf-documents
```
go tool pprof cpuprofile.prof
(pprof) pdf
(pprof) quit

// If the "pdf"-command not work, install: www.graphviz.org 

go tool pprof memprofile.prof
(pprof) pdf
(pprof) quit
´´´
