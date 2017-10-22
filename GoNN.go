// ownNN
package main

import (
	"fmt"
	DSP "goNN/dataset_port"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

// Confic the Neural Network
var mnistNNconfig NNconfig = NNconfig{InputNodes: 784, HiddenNodes: 200, OutputNodes: 10, Epochs: 1, learningRate: 0.2}

// Cinfig the Data Store
var mnistDSconfig DSP.DSconfig = DSP.DSconfig{InputData: 784, OutputData: 1, PathName: "mnist_dataset/", TrainingFileName: "mnist_train.csv",
	ValidationFileName: "", TestFileName: "mnist_test.csv"}

//var mnistDSconfig DSP.DSconfig = DSP.DSconfig{InputData: 784, OutputData: 1, PathName: "mnist_dataset/", TrainingFileName: "mnist_train_100.csv",
//	ValidationFileName: "", TestFileName: "mnist_test_10.csv"}

type NNconfig struct { // to configurate a Neural Network
	InputNodes   int
	HiddenNodes  int
	OutputNodes  int
	Epochs       int
	learningRate float64
}

type NeuralNetwork struct {
	inputNodes  int // = InputData
	hiddenNodes int // = 200
	outputNodes int //= 10

	epochs       int     // = 1
	learningRate float64 // 0.1
	// Input
	inputVector []float64 // input-data transformed to 0.1 ... 0.99

	// Hidden
	hiddenInVector    []float64 // the sum of all income weight * inputVector
	hiddenOutVector   []float64 // sigmoid of hiddenInVector
	hiddenErrorVector []float64 // the sum of all outcome wight * outputError
	// Output
	outputInVector     []float64 // the sum of all income weight * hiddenOutVector
	outputOutVector    []float64 // sigmoid of all outputInVector
	outputErrorVector  []float64
	outputTargetVector []float64 // output-data from file transfomed to 0.1 ... 0.99

	wih [][]float64 // [HiddenNodes][InputNodes]
	who [][]float64 // [OutputNodes][HiddenNodes]
	//	wihAnf    [][]float64 // [HiddenNodes][InputNodes]
	//	whoAnf    [][]float64 // [OutputNodes][HiddenNodes]
	startTime time.Time //  to initialize the pseudo-random number generator
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

// the sigmoid-function
func sig(x float64) float64 {
	ehx := math.Exp(x)
	return ehx / (1.0 + ehx)
}

func main() {
	// CPU-Profile
	cpu_f, cpu_err := os.Create("cpuprofile.prof")
	if cpu_err != nil {
		log.Fatal("could not create CPU profile: ", cpu_err)
	}
	if p_err := pprof.StartCPUProfile(cpu_f); p_err != nil {
		log.Fatal("could not start CPU profile: ", p_err)
	}
	defer pprof.StopCPUProfile()
	// END CPU-Profile

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

	// memory-profile
	mem_f, mem_err := os.Create("memprofile.prof")
	if mem_err != nil {
		log.Fatal("could not create memory profile: ", mem_err)
	}
	runtime.GC() // get up-to-date statistics
	if mem_err := pprof.WriteHeapProfile(mem_f); mem_err != nil {
		log.Fatal("could not write memory profile: ", mem_err)
	}
	mem_f.Close()
	// END memory-profile
}

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

func (nn *NeuralNetwork) InitNeuralNetwork(nnConfig NNconfig) {
	inputNodes := nnConfig.InputNodes
	hiddenNodes := nnConfig.HiddenNodes
	outputNodes := nnConfig.OutputNodes

	nn.inputNodes = inputNodes
	nn.hiddenNodes = hiddenNodes
	nn.outputNodes = outputNodes

	nn.inputVector = make([]float64, inputNodes) // input-data transformed to 0.1 ... 0.99

	nn.outputTargetVector = make([]float64, outputNodes) // output-data from file transfomed to 0.1 ... 0.99
	// Hidden
	nn.hiddenInVector = make([]float64, hiddenNodes)    // the sum of all income weight * inputVector
	nn.hiddenOutVector = make([]float64, hiddenNodes)   // sigmoid of hiddenInVector
	nn.hiddenErrorVector = make([]float64, hiddenNodes) // the sum of all outcome wight * outputError
	// Output
	nn.outputInVector = make([]float64, outputNodes)  // the sum of all income weight * hiddenOutVector
	nn.outputOutVector = make([]float64, outputNodes) // sigmoid of all outputInVector
	nn.outputErrorVector = make([]float64, outputNodes)

	nn.startTime = time.Now() //  to initialize the pseudo-random number generator

	nn.learningRate = nnConfig.learningRate
	nn.epochs = nnConfig.Epochs

	// init wights
	//  first init r start of random-generator
	rand.Seed(int64(nn.startTime.Nanosecond()) / 1000)
	// r := rand.New(rand.NewSource(int64(cnn.startTime.Nanosecond()) / 1000))
	// init weight input-hidden with r.Floar64 - 0.5
	nn.wih = make([][]float64, nn.hiddenNodes)
	for h := 0; h < nn.hiddenNodes; h++ {
		nn.wih[h] = make([]float64, nn.inputNodes)
		for i := 0; i < nn.inputNodes; i++ {
			nn.wih[h][i] = rand.Float64() - 0.5
			// cnn.wih[h][i] = r.Float64() - 0.5
			// cnn.wihAnf[h][i] = cnn.wih[h][i]
			// fmt.Printf("%f|", wihAnf[h][i])
		}
		// fmt.Println()
	}
	// fmt.Println()
	nn.who = make([][]float64, nn.outputNodes)
	for o := 0; o < nn.outputNodes; o++ {
		nn.who[o] = make([]float64, nn.hiddenNodes)
		for h := 0; h < nn.hiddenNodes; h++ {
			nn.who[o][h] = rand.Float64() - 0.5
			// cnn.who[o][h] = r.Float64() - 0.5
			// cnn.whoAnf[o][h] = cnn.who[o][h]
			// fmt.Printf("%f//", whoAnf[o][h])
		}
		// fmt.Println()
	}
	return
}

func (nn *NeuralNetwork) Train(ds DataSetInterface, epochs int, learningRate float64) (err error) {

	err = ds.OpenTrainingDataFile()

	defer ds.CloseDataFile()

	for epochsCounter := 0; err == nil && epochsCounter < epochs; epochsCounter++ {
		for err = ds.ReadNextDataSet(); err == nil; err = ds.ReadNextDataSet() {

			ds.InputData2InputNodes(nn.inputVector)
			// fmt.Printf(" --- InputVector --- \n")
			//	for _, inData := range nn.inputVector {
			//		fmt.Print(inData, "|")
			//	}
			//	fmt.Println()

			//	fmt.Printf(" --- HiddenInVector ---\n")
			for h, hvec := range nn.wih {
				sum := 0.0
				for i, w := range hvec {
					sum += w * nn.inputVector[i]
				}
				nn.hiddenInVector[h] = sum
				nn.hiddenOutVector[h] = sig(sum)
				//		fmt.Printf("%f[%d],", sum, h)
				//		fmt.Printf("%f[%d],", nn.hiddenOutVector[h], h)
			}

			//	fmt.Printf("\n --- OutputInVector ---- \n")
			for o, ovec := range nn.who {
				sum := 0.0
				for h, w := range ovec {
					sum += w * nn.hiddenOutVector[h]
				}
				nn.outputInVector[o] = sum
				nn.outputOutVector[o] = sig(sum)
				//		fmt.Printf("%f[%d];", nn.outputInVector[o], o)
				//		fmt.Printf("%f[%d];", nn.outputOutVector[o], o)
			}

			ds.OutputData2TargetNodes(nn.outputTargetVector)

			//	fmt.Println("\n ---- Excpected Output ----\n")
			//	for o, oData := range nn.outputTargetVector {
			//		fmt.Printf("%f[%d]:", oData, o)
			//	}
			//	fmt.Println("\n --- Output Errors ---")
			for o := 0; o < nn.outputNodes; o++ {
				nn.outputErrorVector[o] = nn.outputTargetVector[o] - nn.outputOutVector[o]
				//		fmt.Printf("%f[%d]*", nn.outputErrorVector[o], o)
			}
			//	fmt.Println("\n --- Hidden Errors ---")
			for h := 0; h < nn.hiddenNodes; h++ {
				err := 0.0
				for o := 0; o < nn.outputNodes; o++ {
					err += nn.who[o][h] * nn.outputErrorVector[o]
				}
				nn.hiddenErrorVector[h] = err
				//		fmt.Printf("%f[%d];", nn.hiddenErrorVector[h], h)
			}
			//	fmt.Printf("\n --- new h - o weights --- ")
			for h := 0; h < nn.hiddenNodes; h++ {
				for o := 0; o < nn.outputNodes; o++ {
					finalOut := nn.outputOutVector[o]
					diff := learningRate * nn.outputErrorVector[o] * finalOut * (1.0 - finalOut) * nn.hiddenOutVector[h]
					nn.who[o][h] += diff
					//			fmt.Printf("D:%f:W:%f;", diff, who[o][h])
				}
				//		fmt.Println()
			}

			//	fmt.Println("\n -- end --")
			//	fmt.Printf("\n --- new i - h wieghts --- \n")
			for i := 0; i < nn.inputNodes; i++ {
				for h := 0; h < nn.hiddenNodes; h++ {
					finalHidden := nn.hiddenOutVector[h]
					diff := learningRate * nn.hiddenErrorVector[h] * finalHidden * (1.0 - finalHidden) * nn.inputVector[i]
					nn.wih[h][i] += diff
					//			fmt.Printf("D:%f:W:%f;", diff, wih[h][i])
				}
				//		fmt.Println()
			}
		} // for err = readOutputInputData(cnn); err == nil; err = readOutputInputData(cnn) {
		if err == io.EOF {
			err = ds.OpenTrainingDataFile()
		}
	}
	return
}

func (nn *NeuralNetwork) Test(ds DataSetInterface) (result float64, err error) {
	var score, failure int

	err = ds.OpenTestDataFile()

	defer ds.CloseDataFile()

	for err = ds.ReadNextDataSet(); err == nil; err = ds.ReadNextDataSet() {

		ds.InputData2InputNodes(nn.inputVector)
		//  fmt.Printf(" --- inputVector --- \n")
		//	for _, inData := range nn.inputVector {
		//		fmt.Print(inData, "|")
		//	}
		//	fmt.Println()

		//	fmt.Printf(" --- hiddenIn/Out-Vector ---\n")
		for h, hvec := range nn.wih {
			sum := 0.0
			for i, w := range hvec {
				sum += w * nn.inputVector[i]
			}
			nn.hiddenInVector[h] = sum
			nn.hiddenOutVector[h] = sig(sum)
			//		fmt.Printf("hin:%f[%d],", sum, h)
			//		fmt.Printf("hout:%f[%d],", nn.hiddenOutVector[h], h)
		}

		//	fmt.Printf("\n --- OutputIn/Out-Vector ---- \n")
		for o, ovec := range nn.who {
			sum := 0.0
			for h, w := range ovec {
				sum += w * nn.hiddenOutVector[h]
			}
			nn.outputInVector[o] = sum
			nn.outputOutVector[o] = sig(sum)
			//		fmt.Printf("%f[%d];",nn.outputInVector[o], o)
			//		fmt.Printf("%f[%d];", nn.outputOutVector[o], o)
		}

		if ds.AreOutputNotesOK(nn.outputOutVector) {
			score++
		} else {
			failure++
		}
	} // for err = ds.ReadNextDataSet(); err == nil; err = ds.ReadNextDataSet() {
	if err == io.EOF {
		err = nil
		// compute result
		if score > 0 {
			fscore := float64(score)
			result = fscore / (fscore + float64(failure))
		} else {
			result = 0.0
		}
	}
	return
}
