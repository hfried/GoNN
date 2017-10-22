package DataSetPort

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"text/scanner"
)

type DSconfig struct { // to configurate a Data Set
	PathName           string
	TrainingFileName   string
	ValidationFileName string
	TestFileName       string
	InputData          int
	OutputData         int
}

type MNIST_DataSet struct {
	pathName           string
	trainingFileName   string
	validationFileName string
	testFileName       string
	inputData          int
	outputData         int
	inputDataVector    []int
	outputDataVector   []int
	dataFile           *os.File
	curDataFileName    string
}

func isNil(err error) {
	if err != nil {
		panic(fmt.Sprintf("Read_Data MNIST: %v", err))
	}
}

// the validationFileName may be empty (== "")
func (ds *MNIST_DataSet) InitDataSet(dsConfig DSconfig) {
	ds.pathName = dsConfig.PathName
	ds.trainingFileName = dsConfig.TrainingFileName
	ds.validationFileName = dsConfig.ValidationFileName
	ds.testFileName = dsConfig.TestFileName
	ds.inputData = dsConfig.InputData
	ds.outputData = dsConfig.OutputData
	ds.inputDataVector = make([]int, dsConfig.InputData)
	ds.outputDataVector = make([]int, dsConfig.OutputData)
}

func (ds *MNIST_DataSet) openDataFile(fileName string) error {
	dataFile, err := os.Open(ds.pathName + fileName)
	if err != nil {
		dataFile.Close()
		return err
	}
	ds.dataFile = dataFile
	ds.curDataFileName = fileName
	return nil
}

func (ds *MNIST_DataSet) OpenTrainingDataFile() error {
	return ds.openDataFile(ds.trainingFileName)
}

func (ds *MNIST_DataSet) OpenValidationDataFile() error {
	if ds.validationFileName == "" {
		return errors.New(fmt.Sprintf("can not open ValidationDateFile: No validation file name"))
	}
	return ds.openDataFile(ds.validationFileName)
}

func (ds *MNIST_DataSet) OpenTestDataFile() error {
	return ds.openDataFile(ds.testFileName)
}

func (ds *MNIST_DataSet) CloseDataFile() {
	ds.dataFile.Close()
}

func (ds *MNIST_DataSet) ReadNextDataSet() error {

	var line string

	_, err := fmt.Fscanln(ds.dataFile, &line)

	if err != nil {
		ds.dataFile.Close()
		return err
	}

	var s scanner.Scanner
	var intResult int
	var realResult float64
	var strResult string
	var idx int = 0
	var tok rune
	s.Init(strings.NewReader(line))

	s.Filename = ds.curDataFileName

	for tok = s.Scan(); tok != scanner.EOF; tok = s.Scan() {
		// s.Line = i + 1
		strResult = s.TokenText()
		if tok < 0 {
			switch tok {
			case scanner.Int:
				fmt.Sscan(strResult, &intResult)
				// fmt.Printf("i:%d", intResult)
				if idx < ds.outputData {
					ds.outputDataVector[idx] = intResult
					idx++
				} else {
					ds.inputDataVector[idx-ds.outputData] = intResult
					idx++
				}
			case scanner.Float:
				fmt.Sscan(strResult, &realResult)
				fmt.Printf("real:%f", realResult)
				return errors.New(fmt.Sprintf(" Scan float: %s in DataFile", strResult))
			default:
				fmt.Printf("string:%s", strResult)
				return errors.New(fmt.Sprintf("Scan: %s - no int in DataFile", strResult))
			}
		}
	}
	if idx < ds.outputData+ds.inputData {
		return errors.New(fmt.Sprintf("More data expected:  %d Data read, %d Data expected \n", idx, ds.outputData+ds.inputData))
	}

	//	for _, inpData := range ds.inputDataVector {
	//		fmt.Print(inpData, ",")
	//	}
	//	fmt.Println()
	return nil
}

func (ds *MNIST_DataSet) InputData2InputNodes(inputNodesVector []float64) {
	for idx, data := range ds.inputDataVector {
		inputNodesVector[idx] = float64(data)/255.0*0.99 + 0.01
	}
}

func (ds *MNIST_DataSet) OutputData2TargetNodes(targetNodesVector []float64) {
	for idx, _ := range targetNodesVector {
		targetNodesVector[idx] = 0.01
	}
	targetNodesVector[ds.outputDataVector[0]] = 0.99
}

func (ds *MNIST_DataSet) OutputNotes2OutputData(outputNodesVector []float64) {
	max, maxIdx := 0.0, 0
	for o, out := range outputNodesVector {
		if out > max {
			max = out
			maxIdx = o
		}
	}
	ds.outputDataVector[0] = maxIdx
}

func (ds *MNIST_DataSet) AreOutputNotesOK(outputNodesVector []float64) bool {
	max, maxIdx := 0.0, 0
	for o, out := range outputNodesVector {
		if out > max {
			max = out
			maxIdx = o
		}
	}
	if ds.outputDataVector[0] == maxIdx {
		return true
	}
	return false
}

func (ds *MNIST_DataSet) PrintOutputData() {
	fmt.Printf(" Output Data: %d\n", ds.outputDataVector[0])
}
